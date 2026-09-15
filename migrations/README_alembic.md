## Run Alembic Migrations

### Configuration

```bash
cp alembic.ini.example alembic.ini
```

- Update the `alembic.ini` file with your database connection details('sqlalchemy.url').


### (Optional) Create a new migration

```bash
alembic revision --autogenerate -m "Your migration message here"
```

### Run migrations

```bash
alembic upgrade head
```

## References
- [Alembic Documentation](https://alembic.sqlalchemy.org/en/latest/)