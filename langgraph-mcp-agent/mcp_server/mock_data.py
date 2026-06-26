MOCK_FILESYSTEM = {
    "/data/reports/q4_2025.csv": "id,product,revenue\n1,Widget A,50000\n2,Widget B,62000\n3,Widget C,31000",
    "/data/reports/q3_2025.csv": "id,product,revenue\n1,Widget A,45000\n2,Widget B,58000",
    "/data/users.csv": "id,name,email,role\n1,Alice,alice@acme.com,admin\n2,Bob,bob@acme.com,viewer",
    "/data/config.yaml": "app:\n  name: AcmeApp\n  version: 2.1.0\n  debug: false",
}

MOCK_DATABASE = [
    {"id": 1, "name": "Alice", "email": "alice@acme.com", "status": "active", "role": "admin"},
    {"id": 2, "name": "Bob", "email": "bob@acme.com", "status": "inactive", "role": "viewer"},
    {"id": 3, "name": "Charlie", "email": "charlie@acme.com", "status": "active", "role": "editor"},
]
