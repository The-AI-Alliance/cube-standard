#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3>=1.34", "botocore>=1.34"]
# ///
"""
AWS privilege check for cube-infra-aws.

Tests every permission required by AWSInfraConfig without provisioning anything:
  - Read-only describe calls (always safe)
  - ec2:RunInstances with DryRun=True
  - S3 put/delete of a tiny test object (if bucket exists)
  - iam:SimulatePrincipalPolicy for write perms that can't be dry-run

Usage:
    uv run python check_aws_privileges.py
    uv run python check_aws_privileges.py --region us-west-2
"""

from __future__ import annotations

import argparse
import sys
import uuid

import boto3
import botocore.exceptions

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"
INFO = "\033[34m[INFO]\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> bool:
    try:
        msg = fn()
        results.append((name, True, msg or ""))
        print(f"  {PASS} {name}" + (f"  — {msg}" if msg else ""))
        return True
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        # DryRunOperation means we HAVE the permission
        if code == "DryRunOperation":
            results.append((name, True, "DryRun confirmed"))
            print(f"  {PASS} {name}  — DryRun confirmed")
            return True
        results.append((name, False, f"{code}: {msg}"))
        print(f"  {FAIL} {name}  — {code}: {msg}")
        return False
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name}  — {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    session = boto3.session.Session()
    region = args.region or session.region_name
    if not region:
        print(f"{FAIL} No region configured. Set AWS_DEFAULT_REGION or use --region.")
        sys.exit(1)

    print(f"\n{INFO} Region: {region}")

    # ── 1. Identity ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("STS — identity")
    print(f"{'─' * 60}")
    sts = session.client("sts", region_name=region)
    account_id = None

    def check_identity():
        nonlocal account_id
        r = sts.get_caller_identity()
        account_id = r["Account"]
        return f"account={account_id}  arn={r['Arn']}"

    check("sts:GetCallerIdentity", check_identity)

    if not account_id:
        print(f"\n{FAIL} Cannot continue without account ID.")
        sys.exit(1)

    s3_bucket = f"cube-vmimages-{account_id}"
    ec2 = session.client("ec2", region_name=region)
    s3 = session.client("s3", region_name=region)
    iam = session.client("iam")

    # ── 2. EC2 describe ───────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("EC2 — describe (auto-discovery + runtime queries)")
    print(f"{'─' * 60}")

    vpc_id = None

    def check_vpcs():
        nonlocal vpc_id
        r = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
        vpcs = r.get("Vpcs", [])
        if not vpcs:
            return "WARNING: no default VPC — set vpc_id= explicitly"
        vpc_id = vpcs[0]["VpcId"]
        return f"default VPC: {vpc_id}"

    check("ec2:DescribeVpcs", check_vpcs)

    subnet_id = None

    def check_subnets():
        nonlocal subnet_id
        if not vpc_id:
            return "skipped (no VPC)"
        r = ec2.describe_subnets(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "default-for-az", "Values": ["true"]},
            ]
        )
        subnets = r.get("Subnets", [])
        if not subnets:
            return "WARNING: no default subnets"
        subnet_id = subnets[0]["SubnetId"]
        return f"{len(subnets)} default subnet(s), first: {subnet_id}"

    check("ec2:DescribeSubnets", check_subnets)

    check(
        "ec2:DescribeInstances",
        lambda: (
            f"{sum(len(r['Instances']) for r in ec2.describe_instances(Filters=[{'Name': 'tag:project', 'Values': ['cube']}]).get('Reservations', []))} cube instance(s) currently running"
        ),
    )

    # Approved ServiceNow AMI — must be used instead of public images (SCP restriction)
    APPROVED_AMI = "ami-0b4197b68cad72a93"  # ServiceNow_Redhat9_Image_2025061601
    APPROVED_INSTANCE_TYPE = "t2.medium"
    ami_id_for_dryrun = APPROVED_AMI

    def check_images():
        r = ec2.describe_images(ImageIds=[APPROVED_AMI])
        images = r.get("Images", [])
        if not images:
            return f"WARNING: approved AMI {APPROVED_AMI} not visible in this region"
        img = images[0]
        return f"{img['Name']}  state={img['State']}"

    check(f"ec2:DescribeImages (approved AMI {APPROVED_AMI})", check_images)
    check(
        "ec2:DescribeSecurityGroups",
        lambda: (
            f"{len(ec2.describe_security_groups(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]).get('SecurityGroups', []))} SGs in VPC"
            if vpc_id
            else "skipped"
        ),
    )
    check(
        "ec2:DescribeKeyPairs",
        lambda: f"{len(ec2.describe_key_pairs().get('KeyPairs', []))} key pair(s)",
    )
    check(
        "ec2:DescribeImportSnapshotTasks",
        lambda: f"{len(ec2.describe_import_snapshot_tasks().get('ImportSnapshotTasks', []))} task(s)",
    )
    check(
        "ec2:DescribeSnapshots",
        lambda: f"{len(ec2.describe_snapshots(OwnerIds=['self']).get('Snapshots', []))} snapshot(s) owned",
    )

    # ── 3. EC2 write (DryRun) ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("EC2 — write operations (DryRun=True, no resources created)")
    print(f"{'─' * 60}")

    sg_id_for_dryrun = None
    try:
        sgs = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}] if vpc_id else []).get(
            "SecurityGroups", []
        )
        if sgs:
            sg_id_for_dryrun = sgs[0]["GroupId"]
    except Exception:
        pass

    def dryrun_run_instances():
        if not subnet_id or not sg_id_for_dryrun:
            return "skipped (missing subnet/sg from describe checks)"
        ec2.run_instances(
            ImageId=ami_id_for_dryrun,
            InstanceType=APPROVED_INSTANCE_TYPE,
            MinCount=1,
            MaxCount=1,
            DryRun=True,
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": subnet_id,
                    "Groups": [sg_id_for_dryrun],
                    "AssociatePublicIpAddress": True,
                }
            ],
        )

    check("ec2:RunInstances (DryRun)", dryrun_run_instances)

    def dryrun_terminate():
        # Use a fake instance ID — TerminateInstances with DryRun raises DryRunOperation
        # before it validates the instance ID
        ec2.terminate_instances(InstanceIds=["i-00000000000000000"], DryRun=True)

    check("ec2:TerminateInstances (DryRun)", dryrun_terminate)

    def dryrun_create_sg():
        if not vpc_id:
            return "skipped"
        ec2.create_security_group(
            GroupName=f"cube-dryrun-{uuid.uuid4().hex[:6]}",
            Description="DryRun test",
            VpcId=vpc_id,
            DryRun=True,
        )

    check("ec2:CreateSecurityGroup (DryRun)", dryrun_create_sg)

    def dryrun_authorize_ingress():
        if not sg_id_for_dryrun:
            return "skipped"
        ec2.authorize_security_group_ingress(
            GroupId=sg_id_for_dryrun,
            IpProtocol="tcp",
            FromPort=22,
            ToPort=22,
            CidrIp="0.0.0.0/0",
            DryRun=True,
        )

    check("ec2:AuthorizeSecurityGroupIngress (DryRun)", dryrun_authorize_ingress)

    def dryrun_import_key_pair():
        import base64

        ec2.import_key_pair(
            KeyName=f"cube-dryrun-{uuid.uuid4().hex[:6]}",
            PublicKeyMaterial=base64.b64encode(b"ssh-ed25519 AAAAB placeholder"),
            DryRun=True,
        )

    check("ec2:ImportKeyPair (DryRun)", dryrun_import_key_pair)

    def dryrun_register_image():
        ec2.register_image(
            Name=f"cube-dryrun-{uuid.uuid4().hex[:6]}",
            Architecture="x86_64",
            RootDeviceName="/dev/sda1",
            VirtualizationType="hvm",
            DryRun=True,
            BlockDeviceMappings=[],
        )

    check("ec2:RegisterImage (DryRun)", dryrun_register_image)

    def dryrun_deregister_image():
        # Fake AMI ID — DryRun fires before validation
        ec2.deregister_image(ImageId="ami-00000000000000000", DryRun=True)

    check("ec2:DeregisterImage (DryRun)", dryrun_deregister_image)

    def dryrun_delete_snapshot():
        ec2.delete_snapshot(SnapshotId="snap-00000000000000000", DryRun=True)

    check("ec2:DeleteSnapshot (DryRun)", dryrun_delete_snapshot)

    # ImportSnapshot has no DryRun — use SimulatePrincipalPolicy below

    # ── 4. S3 ─────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"S3 — bucket: {s3_bucket}")
    print(f"{'─' * 60}")

    bucket_exists = False

    def check_s3_head():
        nonlocal bucket_exists
        try:
            s3.head_bucket(Bucket=s3_bucket)
            bucket_exists = True
            return "bucket exists"
        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                return "bucket does not exist yet (will be created by provision())"
            raise

    check("s3:HeadBucket / s3:GetBucketLocation", check_s3_head)

    def check_s3_list():
        r = s3.list_buckets()
        names = [b["Name"] for b in r.get("Buckets", [])]
        cube_buckets = [n for n in names if "cube" in n]
        return f"{len(names)} bucket(s) total, cube-related: {cube_buckets or 'none'}"

    check("s3:ListBuckets", check_s3_list)

    if bucket_exists:
        test_key = f"_privilege_check_{uuid.uuid4().hex}.txt"

        def check_s3_put():
            s3.put_object(Bucket=s3_bucket, Key=test_key, Body=b"cube privilege check")
            return f"wrote s3://{s3_bucket}/{test_key}"

        ok = check("s3:PutObject", check_s3_put)

        if ok:

            def check_s3_get():
                r = s3.get_object(Bucket=s3_bucket, Key=test_key)
                return f"{len(r['Body'].read())} bytes"

            check("s3:GetObject", check_s3_get)

            def check_s3_delete():
                s3.delete_object(Bucket=s3_bucket, Key=test_key)
                return f"deleted s3://{s3_bucket}/{test_key}"

            check("s3:DeleteObject", check_s3_delete)
    else:
        print(f"  {SKIP} s3:PutObject / GetObject / DeleteObject  — bucket not yet created")
        print("         (these will be tested implicitly when provision() creates the bucket)")

    # ── 5. IAM — read ────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("IAM — read (roles & instance profiles)")
    print(f"{'─' * 60}")

    caller_arn = None
    try:
        caller_arn = sts.get_caller_identity()["Arn"]
    except Exception:
        pass

    def check_iam_get_vmimport():
        try:
            r = iam.get_role(RoleName="vmimport")
            return f"role exists: {r['Role']['Arn']}"
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                return "role does not exist yet (will be created by provision())"
            raise

    check("iam:GetRole (vmimport)", check_iam_get_vmimport)

    def check_iam_get_bootstrap_role():
        try:
            r = iam.get_role(RoleName="cube-bootstrap-role")
            return f"role exists: {r['Role']['Arn']}"
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                return "role does not exist yet (will be created by provision())"
            raise

    check("iam:GetRole (cube-bootstrap-role)", check_iam_get_bootstrap_role)

    def check_iam_get_profile():
        try:
            r = iam.get_instance_profile(InstanceProfileName="cube-bootstrap")
            return f"profile exists: {r['InstanceProfile']['Arn']}"
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                return "profile does not exist yet (will be created by provision())"
            raise

    check("iam:GetInstanceProfile (cube-bootstrap)", check_iam_get_profile)

    # ── 6. IAM — SimulatePrincipalPolicy for write ops ───────────────────────
    print(f"\n{'─' * 60}")
    print("IAM — SimulatePrincipalPolicy (write permissions we can't DryRun)")
    print(f"{'─' * 60}")

    write_actions = [
        "s3:CreateBucket",
        "s3:PutBucketPolicy",
        "ec2:ImportSnapshot",
        "iam:CreateRole",
        "iam:PutRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:PassRole",
    ]

    def check_simulate():
        if not caller_arn:
            return "skipped (could not determine caller ARN)"
        r = iam.simulate_principal_policy(
            PolicySourceArn=caller_arn,
            ActionNames=write_actions,
            ResourceArns=["*"],
        )
        denied = [res["EvalActionName"] for res in r.get("EvaluationResults", []) if res["EvalDecision"] != "allowed"]
        if denied:
            raise PermissionError(f"denied: {', '.join(denied)}")
        return f"all {len(write_actions)} actions allowed"

    sim_ok = check("iam:SimulatePrincipalPolicy", check_simulate)

    if not sim_ok:
        print(f"  {INFO} SimulatePrincipalPolicy itself may be denied — checking individually:")
        for action in write_actions:

            def _sim(a=action):
                if not caller_arn:
                    return "skipped"
                r = iam.simulate_principal_policy(
                    PolicySourceArn=caller_arn,
                    ActionNames=[a],
                    ResourceArns=["*"],
                )
                result = r["EvaluationResults"][0]["EvalDecision"]
                if result != "allowed":
                    raise PermissionError(f"decision: {result}")
                return "allowed"

            check(f"  simulate: {action}", _sim)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print(f"\n{FAIL} Missing permissions:")
        for name, ok, msg in results:
            if not ok:
                print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"{PASS} All checks passed — account looks ready for AWSInfraConfig.")
    print()


if __name__ == "__main__":
    main()
