"""Builds data/eval_lost.csv and data/eval_found.csv (synthetic, hand-written test set)."""
import csv, os
os.makedirs("data", exist_ok=True)

# (item, description, location) lost  ->  found paraphrase (different wording on purpose)
pairs = [
 (("water bottle","black water bottle with a sticker","library"),
  ("bottle","dark coloured bottle with a label","outside the library")),
 (("wallet","brown leather wallet with college id inside","canteen"),
  ("purse","tan leather wallet containing a student card","near the canteen")),
 (("phone","blue Samsung phone with cracked screen","main block"),
  ("mobile","blue samsung smartphone with a crack on the display","main building")),
 (("laptop","silver HP laptop with a sticker on the lid","lab 3"),
  ("laptop","grey hp notebook computer with stickers on cover","computer lab 3")),
 (("backpack","red Wildcraft backpack with a keychain","bus stop"),
  ("bag","red wildcraft rucksack with key chain attached","near bus stop")),
 (("earphones","white boat earphones in a black case","seminar hall"),
  ("earbuds","white boat earbuds with dark case","seminar hall")),
 (("id card","student id card with blue lanyard","parking area"),
  ("identity card","college id with blue strap","parking")),
 (("umbrella","green folding umbrella","admin block"),
  ("umbrella","green foldable umbrella","near admin block")),
 (("keys","bunch of keys with a red keychain","hostel gate"),
  ("key","set of keys with red key ring","hostel entrance")),
 (("watch","black Casio digital watch","gym"),
  ("wristwatch","dark casio digital wrist watch","gymnasium")),
 (("glasses","black framed spectacles in a brown case","library"),
  ("spectacles","dark frame glasses with brown case","library reading hall")),
 (("charger","white Apple charger with cable","cafeteria"),
  ("adapter","white apple adapter and cable","cafeteria")),
 (("book","blue data structures textbook with name written","class 204"),
  ("textbook","blue book on data structures with a name on it","classroom 204")),
 (("calculator","grey Casio scientific calculator","exam hall"),
  ("calculator","silver casio scientific calculator","examination hall")),
 (("pen drive","red 32GB pen drive","computer lab"),
  ("usb drive","red 32gb flash drive","computer lab")),
 (("power bank","black Redmi power bank with scratches","bus stop"),
  ("powerbank","dark redmi power bank scratched","bus stop")),
 (("bottle","blue Milton bottle with dent","sports ground"),
  ("flask","blue milton flask with a dent","ground")),
 (("handbag","pink handbag with a name tag","auditorium"),
  ("bag","pink handbag with name tag","auditorium")),
 (("headphones","black Sony headphones","library"),
  ("headset","dark sony headphones","reading room library")),
 (("smartwatch","black Fastrack smartwatch with strap","hostel"),
  ("watch","dark fastrack smart watch","hostel corridor")),
]

# extra found items that match nobody (including hard negatives)
distractors = [
 ("bottle","red water bottle with a sticker","library"),               # same item+place, other colour
 ("wallet","black wallet with cards","canteen"),                       # same item, other colour
 ("phone","black oneplus phone","main block"),                         # same item, other brand
 ("laptop","black dell laptop","lab 3"),
 ("bag","blue skybags backpack","bus stop"),
 ("umbrella","black umbrella","gate"),
 ("keys","single silver key","parking"),
 ("book","yellow physics notebook","class 101"),
 ("id card","id card with red lanyard","admin block"),
 ("earphones","black jbl earphones","gym"),
 ("watch","gold titan analog watch","cafeteria"),
 ("glasses","sunglasses with white frame","sports ground"),
 ("charger","black samsung charger","hostel"),
 ("calculator","black calculator","library"),
]

# lost reports with NO true match in the found pool
no_match = [
 ("laptop","white Asus laptop","library"),
 ("guitar","brown acoustic guitar","auditorium"),
 ("jacket","grey hoodie with zip","gym"),
 ("tablet","silver Samsung tablet","seminar hall"),
 ("bicycle","green cycle with basket","parking"),
 ("lunch box","orange lunch box","canteen"),
]

with open("data/eval_found.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","type","item","description","location","contact"])
    n = 1
    for i,(l,fo) in enumerate(pairs):
        w.writerow([f"F{i+1}","found",*fo,"finder@x.edu"])
    for j,d in enumerate(distractors):
        w.writerow([f"F{len(pairs)+j+1}","found",*d,"finder@x.edu"])

with open("data/eval_lost.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","type","item","description","location","contact","true_match"])
    for i,(l,fo) in enumerate(pairs):
        w.writerow([f"L{i+1}","lost",*l,"owner@x.edu",f"F{i+1}"])
    for k,d in enumerate(no_match):
        w.writerow([f"L{len(pairs)+k+1}","lost",*d,"owner@x.edu",""])
print("pairs",len(pairs),"found pool",len(pairs)+len(distractors),"lost queries",len(pairs)+len(no_match))
