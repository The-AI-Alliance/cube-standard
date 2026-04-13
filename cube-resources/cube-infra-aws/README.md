# cube-infra-aws

AWS `InfraConfig` implementation for the CUBE resource lifecycle.

This package provides `AWSInfraConfig`, which provisions VM images as EC2 AMIs and
launches short-lived task instances from them.

---

## Prerequisites

Most AWS resources are created automatically by `AWSInfraConfig` on first use (S3 bucket,
IAM roles, security group, key pair). You only need to ensure:

1. **AWS credentials** are configured via the boto3 default chain:
   - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)
   - `~/.aws/credentials` (via `aws configure`)
   - IAM instance profile (if running on EC2)

   ```bash
   aws configure
   ```

2. **IAM permissions**: your credentials must allow EC2, S3, and IAM operations.
   At minimum: `ec2:*`, `s3:*`, `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:PassRole`.

3. **VM Import service role**: the EC2 VM Import pipeline requires a specific IAM role
   named `vmimport`. `AWSInfraConfig` creates this automatically on first provision,
   but your credentials must have `iam:CreateRole` and `iam:PutRolePolicy` permissions.

---

## Listing existing resources

If you're joining a team that already has an AWS setup:

```bash
# Find your default region
aws configure get region

# Find S3 buckets used for VM images (named cube-vmimages-{account_id})
aws s3 ls | grep cube-vmimages

# Find existing AMIs provisioned by the team
aws ec2 describe-images --owners self \
    --filters "Name=name,Values=osworld-ubuntu-vm*" \
    --query 'Images[*].{ID:ImageId,Name:Name,State:State}' \
    --output table

# Find running instances (tagged cube=true)
aws ec2 describe-instances \
    --filters "Name=tag-key,Values=cube" \
    --query 'Reservations[*].Instances[*].{ID:InstanceId,State:State.Name,Launch:LaunchTime}' \
    --output table
```

---

## Quick start

```python
from cube_infra_aws import AWSInfraConfig
from osworld_cube.task import OSWORLD_UBUNTU_RESOURCE

# Minimal — region + VPC + S3 bucket all auto-discovered
infra = AWSInfraConfig()

# With explicit region:
# infra = AWSInfraConfig(region="us-west-2")

# First time only (~30-90 min): downloads qcow2, imports as AMI
infra.provision(OSWORLD_UBUNTU_RESOURCE)

# Every subsequent call: instant (reads from ProvisionStore cache)
infra.provision(OSWORLD_UBUNTU_RESOURCE)

# Launch an instance (~3-5 min): creates EC2 instance, opens SSH tunnel
handle = infra.launch(OSWORLD_UBUNTU_RESOURCE)
print(handle.endpoint)   # http://localhost:<port>
handle.close()           # terminates instance
```

If a team member has already provisioned the AMI, you can skip `provision()` by
registering the existing AMI in your local ProvisionStore:

```python
from cube.provision_store import ProvisionStore

ProvisionStore().put(OSWORLD_UBUNTU_RESOURCE, infra, {
    "ami_id": "ami-0123456789abcdef0",
})
```

> Note: the SSH key baked into the shared AMI must match your `ssh_privkey_path`.
> See [cube-standard#78](https://github.com/The-AI-Alliance/cube-standard/issues/78)
> for the long-term plan to make images key-agnostic.

---

## Integration test

Provisions a fresh `-test` AMI, runs a debug episode, then unprovisions:

```bash
cd cube-resources/cube-infra-aws
uv run python test_run_debug_agent.py
```

Expected runtime: ~45 min (30-90 min provision + 5 min episode).
