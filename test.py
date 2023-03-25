import requests
 
data = {'text':'我就要在中国传媒大学吃上崔永元真面','model':'rbt3-mlp-ner'} 
r = requests.get("http://127.0.0.1:5000/upload", params=data)
 
print(r.text,r.raw,r.status_code,r.content)


# from spacy import displacy
# ex = {'text': '我就要在中国传媒大学吃上崔永元真面', 'ents': [{'label': 'GPE.NAM', 'start': 4, 'end': 5}, {'label': 'ORG.NAM', 'start': 6, 'end': 9}, {'label': 'PER.NAM', 'start': 12, 'end': 15}], 'title': None}
# html = displacy.render(ex, style="ent", manual=True)
# print(html) 