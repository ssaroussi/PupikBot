import requests
from bs4 import BeautifulSoup

url = "http://www.mizrahit.co/"
f_name = 'songs.dat'

def getPageText(id):
    return requests.get(url + 'lyrics.php?id={}'.format(id)).text

def getLyricsPage(id):
    page = getPageText(id)

    if ('שיר זה עדיין לא אושר על ידי המערכת.' in page):
        return None
        
    return ''.join(map(str, (BeautifulSoup(page, 'html.parser').find("div", {"id": "songText"}).contents))).replace('<br/>', '')

f = open(f_name, 'a', encoding='utf-8')
valid_songs = 0

for i in range(41, 10000):

    tex = getLyricsPage(i)
    if tex != None:
        valid_songs += 1
        f.write(tex)

    print ('Attempts: {}\tValids: {}'.format(i, valid_songs))

f.close()
