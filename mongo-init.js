db.createUser(
    {
    user: "Admin",
    pwd: "123456",
    roles: [ { role: "userAdminAnyDatabase", db: "admin" } ]
    }
)