'''
sqlite can use a database without a server. sqlite file is not like a flat file. if you want to load
a large file/table. you cant load it into the entire buffer.
'''
import sqlite3
import time
import datetime
import random

conn = sqlite3.connect('tutorial.db')
c =conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS stufftoplot(unix REAL, datestamp TEXT, keyword TEXT, value REAL)')
    #caps for sql command
    
def data_entry():
    c.execute("INSERT INTO stufftoplot VALUES(14233645,'07-12-2017', 'Pysthon', 10)")
    conn.commit() #always needs to be done when value is inseerted
    c.close() #only at the end
    conn.close() #only at the end

def dynamic_data_entry():
    unix=time.time()
    date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))
    keyword = 'Python'
    value = random.randrange(0,10)
    c.execute("INSERT INTO stufftoplot (unix, datestamp, keyword, value) VALUES(?, ?, ?, ?)",(unix, date, keyword, value))
    #in sqlite you use ? , but in mysql we use 5s or %d etc.
    conn.commit()

def read_from_db():
##    c.execute("SELECT * FROM stufftoplot GROUP BY value")
    c.execute("SELECT keyword, unix, datestamp FROM stufftoplot WHERE unix> 1245543")
##    data = c.fetchall()
##    print(data)
    for row in c.fetchall():
        print(row)



create_table()
##data_entry()
#used to enter data
##for i in range(10):
##    dynamic_data_entry()
##    time.sleep(1)

read_from_db()

c.close()
conn.close()
