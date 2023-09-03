import sqlite3


def camera_urls_from_db():
    connection = sqlite3.connect("../db.sqlite3")
    cursor = connection.cursor()
    query = cursor.execute("""SELECT url FROM 'api_camera';""")
    urls = [i[0] for i in query.fetchall()]
    return urls


camera_urls = camera_urls_from_db()

if __name__ == "__main__":
    print(camera_urls)