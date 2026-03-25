"""
Resource tracker for Azure experiment resources.

Every resource we create gets:
1. Tagged with project=cube-experiment in Azure
2. Recorded here in resources.json

To clean up everything: python track.py delete
To list what we have: python track.py list
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MANIFEST = Path(__file__).parent / "resources.json"
TAG = "project=cube-experiment"


def load() -> dict:
    return json.loads(MANIFEST.read_text())


def save(data: dict):
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


def record(name: str, resource_type: str, resource_id: str):
    """Call this after creating any Azure resource."""
    data = load()
    data["resources"].append({
        "name": name,
        "type": resource_type,
        "id": resource_id,
        "created_at": datetime.utcnow().isoformat(),
    })
    save(data)
    print(f"[track] Recorded: {resource_type} / {name}")


def list_resources():
    data = load()
    if not data["resources"]:
        print("No tracked resources.")
        return
    print(f"\n{'Name':<40} {'Type':<35} {'Created'}")
    print("-" * 90)
    for r in data["resources"]:
        print(f"{r['name']:<40} {r['type']:<35} {r['created_at'][:19]}")

    # Also query Azure directly by tag to catch anything missed
    print(f"\n--- Azure query (tag: {TAG}) ---")
    result = subprocess.run(
        ["az", "resource", "list", "--tag", TAG,
         "--subscription", data["subscription"],
         "--query", "[].{name:name, type:type, id:id}", "-o", "table"],
        capture_output=False,
    )


def delete_all():
    data = load()
    if not data["resources"]:
        print("Nothing to delete.")
        return

    print(f"Will delete {len(data['resources'])} tracked resources:")
    list_resources()
    if "--yes" not in sys.argv:
        confirm = input("\nType 'yes' to confirm deletion: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    # Delete by tag (catches anything not in manifest too)
    print("\nDeleting all resources tagged project=cube-experiment ...")
    result = subprocess.run(
        ["az", "resource", "list", "--tag", TAG,
         "--subscription", data["subscription"],
         "--query", "[].id", "-o", "tsv"],
        capture_output=True, text=True,
    )
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for rid in ids:
        print(f"  Deleting {rid}")
        subprocess.run(["az", "resource", "delete", "--ids", rid], check=True)

    data["resources"] = []
    save(data)
    print("Done. Manifest cleared.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        list_resources()
    elif cmd == "delete":
        delete_all()
    else:
        print(f"Unknown command: {cmd}. Use 'list' or 'delete'.")
