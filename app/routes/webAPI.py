from flask import Blueprint, request,jsonify
from app.webQA import WebQA

WebQA_blueprint = Blueprint('webqa',__name__)

model_path = 'app/model/wangchanberta'
url_path = "https://www.dataxet.co/media-landscape/2024-th"
sentenceEmbeddingModel='intfloat/multilingual-e5-base'

webqa = WebQA(model=model_path, tokenizer=model_path, embedding_model_name=sentenceEmbeddingModel, url = url_path)

@WebQA_blueprint.route('/api/webqa', methods=['POST'])
def get_webqa():
    data = request.get_json()
    user_message = data['message']

    response = webqa.chat_interface(user_message)
    return jsonify({'similar_contexts': response})


