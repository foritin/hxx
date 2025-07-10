from vnpy.trader.database import get_database


db = get_database()
overview = db.get_bar_overview()
print()
