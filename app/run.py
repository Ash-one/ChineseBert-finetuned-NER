from flask import Flask, request, make_response
from flask import Response, render_template
import loader
from model import BertNER,BertBilstmCrfNER
from spacy import displacy


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.get('/predict')
def predict__text():
    text = request.args.get('text')
    model_name = request.args.get('model')
    print('text:',text,'model_name:',model_name)
    # 下面对不合法或无法读取的模型进行处理
    try:
        model,label_idx_list = loader.load_model_from_file(model_name)
    except IOError:
        return 'IOError'
    if model == None or label_idx_list == None:
        return 'Model not found'
    
    result = {}
    result['text'] = text
    numbers = [int(x) for x in loader.predict(text,model)]
    

    labels = loader.convert_result2label(numbers,label_idx_list)
    entities = loader.convert_label2entity(labels)

    result['ents'] = entities
    result['title'] = None
    print(result)
    
    # 配置需要展示的实体，从loader中加载颜色配置
    options = {"ents": ["ORG.NAM",
                        "GPE.NAM",
                        "PER.NAM",
                        "LOC.NAM",
                        "LOC.NOM",
                        "PER.NOM",
                        "ORG.NOM"
                        ], "colors": loader.load_colors_preset()}
    html = displacy.render(result, style="ent", manual=True, options=options)

    response = make_response(html,200)
    return response



if __name__ == '__main__':
    app.run(debug=True)