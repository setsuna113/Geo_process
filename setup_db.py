from src.database.connection import DatabaseManager
from src.database.setup import DatabaseSetup
from src.config.config import Config

config = Config()
db = DatabaseManager()
setup = DatabaseSetup(db, config)
setup.create_all_tables()
print("✅ Database schema created")