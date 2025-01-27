import aiohttp
import asyncio
import uvicorn
import torch

from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import json

export_file_path = Path(__file__).parent / 'models'

classes = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
# app.mount('/static', StaticFiles(directory='app/static'))

def tensor_to_serializable(tensor):
    return tensor.tolist()

def format_probabilities(probs_tensor):
       # Convert tensor to list and multiply by 100 to get percentages
    probabilities = [prob * 100 for prob in tensor_to_serializable(probs_tensor)]

    # Mapping class indices to provided names
    classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    # Ensure the length of classes matches the length of probabilities
    if len(probabilities) != len(classes):
        raise ValueError("The number of probabilities does not match the number of classes.")

    formatted_probs = {classes[i]: prob for i, prob in enumerate(probabilities)}

    # Optionally, filter to show top N classes
    top_n = 38
    top_classes = sorted(formatted_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return dict(top_classes)



async def setup_learner():
    try:
        
        learn_resnet = load_learner(export_file_path, 'resnet.pkl')
        # torch.nn.Module.dump_patches = True
        learn_densenet = load_learner(export_file_path, 'alexnet.pkl')
        learn_vgg = load_learner(export_file_path, 'vgg.pkl')
        return learn_resnet, learn_densenet, learn_vgg
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = ("\n\nThis model was trained with an old version of fastai and will not work in a CPU environment."
                       "\n\nPlease update the fastai library in your training environment and export your model again."
                       "\n\nSee instructions for 'Returning to work' at https://course.fast.ai.")
            raise RuntimeError(message)
        else:
            raise

async def main():
    return await setup_learner()

loop = asyncio.get_event_loop()
tasks = asyncio.ensure_future(main())
learners = loop.run_until_complete(tasks)
learn_resnet, learn_densenet, learn_vgg = learners




@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction_resnet,_1,probs1 = learn_resnet.predict(img)
    prediction_densenet,_2,probs2 = learn_densenet.predict(img)
    prediction_vgg,_3,probs3 = learn_vgg.predict(img)
    # pred,_, probs = learn.predict(img)

    # Get the predicted class index
    predicted_class_index_1 = torch.argmax(probs1).item()
    predicted_class_index_2 = torch.argmax(probs2).item()
    predicted_class_index_3 = torch.argmax(probs3).item()

# Ensemble by averaging probabilities
    final_probs = (probs1 + probs2 + probs3) / 3

# Get the predicted class index based on the ensemble probabilities
    predicted_class_index = torch.argmax(final_probs).item()

# Get the class name corresponding to the predicted index
    predicted_class = classes[predicted_class_index]








    print(probs1)
    return JSONResponse(
        {   'result': str(prediction_resnet),
            'prob':probs1[predicted_class_index_1].item() * 100,
            'ensampled_pred':str(predicted_class),
            'ensemble_probs': final_probs[predicted_class_index].item()*100,
            'pred':str(prediction_resnet),
                'overview':{
                    'resnet_prediction':str(prediction_resnet),
                    'alexnet_result': str(prediction_densenet),
                    'vgg_result':str(prediction_vgg),
                    'alextnet_prob':probs1[predicted_class_index_1].item() * 100,
                    'densenet_prob':probs2[predicted_class_index_2].item() * 100,
                    'vgg_prob':probs3[predicted_class_index_3].item() * 100,
                    'ensampled_pred':str(predicted_class),
            'ensemble_probs': final_probs[predicted_class_index].item()*100,
                    'auxilary':{
                            'resnet':format_probabilities(probs1),
                            'densenet':format_probabilities(probs2),
                            'vgg':format_probabilities(probs3),
                            'ensemble': format_probabilities(final_probs),
                        }

                         }
                         
                        
                         })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8080, log_level="info")
