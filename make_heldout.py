"""Held-out set written AFTER the lexicon was frozen (no tuning against it)."""
import csv
pairs = [
 (("thermos","steel thermos flask, silver, has a dent","physics lab"),("flask","silver steel thermos with dent at bottom","physics block")),
 (("tiffin","yellow tiffin carrier","canteen"),("lunch box","yellow tiffin box","canteen area")),
 (("phone","white iPhone with transparent case","bus bay"),("mobile","white apple iphone in clear cover","bus bay")),
 (("bag","navy blue laptop bag","cse department"),("backpack","dark blue laptop backpack","computer science dept")),
 (("headphones","red JBL headphones","study room"),("earphones","red jbl over-ear headphones","study hall")),
 (("wallet","black wallet with 500 rupee note","main gate"),("wallet","black leather wallet with cash","gate")),
 (("scarf","maroon woollen scarf","auditorium"),("muffler","maroon wool muffler","auditorium")),
 (("spectacles","round gold spectacles","corridor"),("glasses","gold round frame glasses","first floor corridor")),
 (("notebook","green record book with name","class 305"),("book","green record notebook, name on cover","room 305")),
 (("charger","black Samsung fast charger","seminar hall"),("adapter","dark samsung charger","seminar hall")),
]
distr = [
 ("flask","blue steel flask","physics lab"),("phone","black iphone","bus bay"),("bag","red laptop bag","cse department"),
 ("headphones","black sony headphones","study room"),("wallet","brown wallet","main gate"),("glasses","black round glasses","corridor"),
 ("charger","white apple charger","seminar hall"),("book","blue record book","class 305"),
]
nomatch = [("camera","black Canon camera","sports ground"),("shoes","white running shoes","gym"),("helmet","red helmet","parking")]
with open("data/heldout_found.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["id","type","item","description","location","contact"])
    for i,(l,fo) in enumerate(pairs): w.writerow([f"F{i+1}","found",*fo,"x"])
    for j,d in enumerate(distr): w.writerow([f"F{len(pairs)+j+1}","found",*d,"x"])
with open("data/heldout_lost.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["id","type","item","description","location","contact","true_match"])
    for i,(l,fo) in enumerate(pairs): w.writerow([f"L{i+1}","lost",*l,"x",f"F{i+1}"])
    for k,d in enumerate(nomatch): w.writerow([f"L{len(pairs)+k+1}","lost",*d,"x",""])
