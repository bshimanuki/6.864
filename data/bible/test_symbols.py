import pickle

a=pickle.load(open('cache/by_book/Romans.pickle','rb'))

b=set()

for chapter in a.values():
    for verse in chapter.values():
        for trans, sent in verse.items():
            if trans == '(SBLG)':
                continue
            for c in sent:
                b.add(c)
            if '+' in sent:
                print(trans, sent)

print(sorted(b))
