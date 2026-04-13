# cube-infra-azure

Azure `InfraConfig` implementation for the CUBE resource lifecycle.

This package provides `AzureInfraConfig`, which provisions VM images into an Azure
Compute Gallery and launches short-lived task VMs from them.

---

## Prerequisites

Before calling `provision()` or `launch()`, the following Azure resources must exist.
They are created once per resource group and shared across all future runs.

### 1. Resource group

```bash
az group create --name <rg> --location westus2
```

Replace `<rg>` with your resource group name (e.g. `cube-rg`). All resources below go into this group.

### 2. Storage account

Used to hold intermediate VHD blobs during provisioning. **Must be in the same region as the resource group.**

```bash
az storage account create \
    --name <storage_account> \
    --resource-group <rg> \
    --location westus2 \
    --sku Standard_LRS \
    --kind StorageV2

az storage container create \
    --name vhds \
    --account-name <storage_account>
```

> If you have multiple storage accounts in the resource group, set `storage_account=` explicitly in `AzureInfraConfig`. Otherwise it is auto-discovered.

### 3. VNet and subnet

```bash
az network vnet create \
    --name <vnet_name> \
    --resource-group <rg> \
    --location westus2 \
    --address-prefix 10.0.0.0/16 \
    --subnet-name default \
    --subnet-prefix 10.0.0.0/24
```

> If you have multiple VNets, set `vnet_name=` and `subnet_name=` explicitly.

### 4. Network Security Group (NSG)

Must allow inbound SSH (port 22) so the harness can open tunnels into task VMs.

```bash
az network nsg create \
    --name <nsg_name> \
    --resource-group <rg>

az network nsg rule create \
    --nsg-name <nsg_name> \
    --resource-group <rg> \
    --name AllowSSH \
    --priority 1000 \
    --protocol Tcp \
    --destination-port-ranges 22 \
    --access Allow \
    --direction Inbound
```

For tighter security, replace `--source-address-prefixes '*'` with your team's IP range.

> If you have multiple NSGs, set `nsg_name=` explicitly.

### 5. Compute Gallery

Stores provisioned image definitions and versions.

```bash
az sig create \
    --gallery-name cube_exp_gallery \
    --resource-group <rg> \
    --location westus2
```

### 6. Bootstrap image definition

The bootstrap pipeline runs a lightweight VM (Ubuntu 22.04 + `qemu-utils` + `azcopy`)
to convert and upload qcow2 images. This VM itself needs a gallery image to launch from.

Create the image definition once in the gallery:

```bash
az sig image-definition create \
    --gallery-name cube_exp_gallery \
    --resource-group <rg> \
    --gallery-image-definition cube-ubuntu-22-04 \
    --publisher cube \
    --offer ubuntu \
    --sku 22-04 \
    --os-type Linux \
    --os-state Specialized \
    --hyper-v-generation V2
```

Then create a version from an existing Ubuntu 22.04 VM or marketplace image.
The easiest approach is to launch a standard Ubuntu 22.04 VM, install `qemu-utils`
and `azcopy`, capture it, and register it as `cube-ubuntu-22-04/1.0.0`.

Alternatively, ask a team member who already has this gallery set up to share the
image with your subscription via [gallery sharing](https://learn.microsoft.com/en-us/azure/virtual-machines/share-gallery).

---

## Listing existing resources

If you're joining a team that already has an Azure setup:

```bash
# Find the resource group
az group list -o table

# Find storage accounts
az storage account list --resource-group <rg> -o table

# Find VNets and subnets
az network vnet list --resource-group <rg> -o table
az network vnet subnet list --resource-group <rg> --vnet-name <vnet_name> -o table

# Find NSGs
az network nsg list --resource-group <rg> -o table

# Find Compute Galleries and image definitions
az sig list --resource-group <rg> -o table
az sig image-definition list \
    --resource-group <rg> \
    --gallery-name cube_exp_gallery \
    -o table
```

---

## Quick start

Once the prerequisites exist:

```python
from cube_infra_azure import AzureInfraConfig
from osworld_cube.task import OSWORLD_UBUNTU_RESOURCE

infra = AzureInfraConfig(
    resource_group="cube-rg",
    # storage_account, vnet_name, subnet_name, nsg_name are auto-discovered
    # if there is only one of each in the resource group
)

# First time only (~40 min): downloads qcow2, converts, uploads, publishes gallery image
infra.provision(OSWORLD_UBUNTU_RESOURCE)

# Every subsequent call: instant (reads from ProvisionStore cache)
infra.provision(OSWORLD_UBUNTU_RESOURCE)

# Launch a VM (~3-5 min): creates VM from gallery image, opens SSH tunnel
handle = infra.launch(OSWORLD_UBUNTU_RESOURCE)
print(handle.endpoint)   # http://localhost:<port>
handle.close()           # stops VM, deletes NIC + public IP
```

If a team member has already provisioned the image, you can skip `provision()` by
registering the existing gallery image in your local ProvisionStore:

```python
from cube.provision_store import ProvisionStore

ProvisionStore().put(OSWORLD_UBUNTU_RESOURCE, infra, {
    "image_def": "osworld-ubuntu-vm",
    "version": "1.0.0",
    "image_id": "/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Compute"
               "/galleries/cube_exp_gallery/images/osworld-ubuntu-vm/versions/1.0.0",
})
```

> Note: the SSH key baked into the shared image must match your `ssh_privkey_path`.
> See [cube-standard#78](https://github.com/The-AI-Alliance/cube-standard/issues/78)
> for the long-term plan to make images key-agnostic.

---

## Integration test

Provisions a fresh `-test` image, runs a debug episode, then unprovisions:

```bash
cd cube-resources/cube-infra-azure
uv run python test_run_debug_agent.py
```

Expected runtime: ~45 min (40 min provision + 5 min episode).
