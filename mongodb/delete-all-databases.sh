#!/bin/bash

# Delete all MongoDB databases
# This script connects to MongoDB and drops all user databases

echo "🗑️  Deleting all MongoDB databases..."
echo ""

mongosh << EOF
// Get list of all databases
const dbs = db.adminCommand({listDatabases: 1}).databases;
const systemDbs = ["admin", "config", "local"];

// Delete user databases only
dbs.forEach(function(database) {
  if (!systemDbs.includes(database.name)) {
    print("Dropping database: " + database.name);
    db.getSiblingDB(database.name).dropDatabase();
  }
});

print("");
print("✅ All user databases deleted!");
print("");
print("Remaining databases:");
db.adminCommand({listDatabases: 1}).databases.forEach(function(db) {
  print("  - " + db.name);
});
EOF

echo ""
echo "Done! 🚀"
