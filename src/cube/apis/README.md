# APIs


## Benchmark-Level API

The Benchmark-Level API defines how evaluation harnesses discover, spawn, and manage task instances. This layer handles shared infrastructure, resource allocation, and task lifecycle management.

## `cube/info`

**Request:**
```json
{
  "method": "cube/info",
  "params": {}
}
```

**Response:**
```json
{
  "name": "...",
  "version": "...",
  "description": "...",
  "authors": ["Jane Researcher", "John Developer"],
  "license": "CC-BY-NC-4.0",
  "requirements": {
    "ram_gb": 16,
    "disk_gb": 50,
    "gpu": false
  },
  "num_tasks": 10,
  "metadata": {
    "...": "..."
  }
}
```

## `cube/tasks`

List available tasks with optional filtering and pagination.

**Request:**
```json
{
  "method": "cube/tasks",
  "params": {
    "offset": 0,
    "limit": 10,
    "filter": {
      "difficulty": "medium",
      "domain": "e-commerce",
      "tags": ["form-filling"]
    }
  }
}
```

**Response:**
```json
{
  "tasks": [
    {
      "id": "shopping-cart-123",
      "description": "Add items to cart and complete checkout",
      "difficulty": "medium",
      "tags": ["e-commerce", "form-filling", "multi-step"],
      "estimated_steps": 15,
      "metadata": {
        "domain": "e-commerce",
        "requires_payment": false
      }
    },
    {
      "id": "product-search-456",
      "description": "Search for specific product and add to wishlist",
      "difficulty": "medium",
      "tags": ["e-commerce", "search"],
      "estimated_steps": 8,
      "metadata": {
        "domain": "e-commerce"
      }
    }
  ],
  "total": 156,
  "offset": 0,
  "limit": 10
}
```

## Task-Level API

The Task-Level API defines how agents interact with individual task instances. It combines the Model Context Protocol (MCP) for action execution with CUBE extensions for evaluation semantics.



