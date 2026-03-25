"""
OSWorld cloud VM management.

Knows about OSWorld (HuggingFace URL, image name, guest user).
Neither azure_backend nor aws_backend imports from here.

CLI:
    python osworld.py create_resources --backend azure
    python osworld.py create_resources --backend aws
    python osworld.py launch --backend azure
    python osworld.py launch --backend aws
    python osworld.py stop --backend azure --vm <name>
    python osworld.py stop --backend aws --vm <instance-id>
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from typing import Union

from _common import configure_logging
from azure_backend import AzureBackend
from aws_backend import AWSBackend

log = logging.getLogger(__name__)

HF_URL = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld"
    "/resolve/main/Ubuntu.qcow2.zip"
)
IMAGE_NAME = "cube-osworld"
IMAGE_VERSION = "1.0.0"
OSWORLD_SSH_USER = "user"


def _make_backend(name: str) -> Union[AzureBackend, AWSBackend]:
    if name == "azure":
        return AzureBackend()
    if name == "aws":
        return AWSBackend()
    raise ValueError(f"Unknown backend: {name!r}")


def create_resources(backend: Union[AzureBackend, AWSBackend]) -> None:
    """One-time: bootstrap OSWorld image from HuggingFace into cloud registry."""
    log.info("create_resources: OSWorld  backend=%s", type(backend).__name__)
    log.info("  source: %s", HF_URL)
    log.info("  image:  %s/%s", IMAGE_NAME, IMAGE_VERSION)
    if isinstance(backend, AWSBackend):
        backend.bootstrap(url=HF_URL, image_name=IMAGE_NAME)
    else:
        backend.bootstrap(url=HF_URL, image_name=IMAGE_NAME, version=IMAGE_VERSION)
    log.info("create_resources: done")


def launch(
    backend: Union[AzureBackend, AWSBackend],
    open_tunnel: bool = True,
) -> dict:
    """Launch OSWorld VM from the registered image."""
    if isinstance(backend, AWSBackend):
        return backend.launch(IMAGE_NAME, ssh_user=OSWORLD_SSH_USER, open_tunnel=open_tunnel)
    return backend.launch(IMAGE_NAME, version=IMAGE_VERSION, open_tunnel=open_tunnel)


def stop(backend: Union[AzureBackend, AWSBackend], resource_id: str) -> None:
    """Stop and delete a running OSWorld VM."""
    backend.stop(resource_id)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="osworld",
        description="OSWorld cloud VM management",
    )
    p.add_argument("--backend", choices=["azure", "aws"], default="azure")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show DEBUG logs (bootstrap progress)")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("create_resources",
                   help="One-time: bootstrap OSWorld image from HuggingFace")

    la = sub.add_parser("launch", help="Launch OSWorld VM from registered image")
    la.add_argument("--no-tunnel", action="store_true")

    st = sub.add_parser("stop", help="Stop and delete a running VM")
    st.add_argument("--vm", required=True, metavar="ID",
                    help="vm_name (Azure) or instance_id (AWS)")

    li = sub.add_parser("list", help="List registered images")  # noqa: F841

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(debug=getattr(args, "verbose", False))
    backend = _make_backend(args.backend)

    if args.cmd == "create_resources":
        create_resources(backend)

    elif args.cmd == "launch":
        result = launch(backend, open_tunnel=not getattr(args, "no_tunnel", False))
        printable = {k: v for k, v in result.items() if k != "tunnel"}
        print(json.dumps(printable, indent=2))
        if result.get("tunnel"):
            log.info("Tunnel open at %s — Ctrl+C to close", result["endpoint"])
            try:
                result["tunnel"].wait()
            except KeyboardInterrupt:
                result["tunnel"].terminate()

    elif args.cmd == "stop":
        stop(backend, args.vm)

    elif args.cmd == "list":
        images = backend.list_images()
        for img in images:
            print(json.dumps(img))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
