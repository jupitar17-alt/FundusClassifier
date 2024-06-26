import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

@st.cache(allow_output_mutation=True)  # Cache model to avoid loading on every run
def load_model():
    return tf.keras.models.load_model('eye_disease_classifier_model.h5')

disclaimer_text = """
**Disclaimer:**

This eye disease classifier is not 100% accurate and should not be used as a substitute for professional medical advice. 
Please consult with a qualified ophthalmologist for accurate diagnosis and treatment. 
Additionally, some eye conditions may overlap.

By clicking "I agree", you acknowledge that you have read and understood this disclaimer.
"""

if "disclaimer_agreed" not in st.session_state:
    st.session_state.disclaimer_agreed = False

if not st.session_state.disclaimer_agreed:
    # Display the disclaimer text
    st.markdown(disclaimer_text)
    if st.button("I agree"):
        st.session_state.disclaimer_agreed = True

if st.session_state.disclaimer_agreed:

    try:
        # Load the model
        model = load_model()
        st.success("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        

    # Define the image preprocessing function
    def preprocess_image(img):
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)
        
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE to the grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(gray_img)
        
        # Convert the equalized grayscale image back to RGB
        equalized_img_rgb = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)
        
        # Resize the image to the input size expected by the model
        img_resized = cv2.resize(equalized_img_rgb, (224, 224))
        
        # Normalize the image to [0, 1]
        img_normalized = img_resized / 255.0
        
        # Expand dimensions to match the input shape of the model
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        return img_expanded

    # Define disease labels
    disease_info = {0:{'name': 'Normal', 'description': 'No signs of eye disease detected. Your eye should be healthy!', 'link': 'https://www.nei.nih.gov/learn-about-eye-health/healthy-vision/keep-your-eyes-healthy'},
                    1:{'name': 'Tessellated fundus', 'description': 'A condition where the retinal pigment epithelium has a tessellated or mosaic appearance. Fundus tesselation has been be associated with various factors such as age, myopia, and age-related macular degeneration, although most commonly with myopia', 'link': 'https://my.clevelandclinic.org/health/diseases/8579-myopia-nearsightedness'},
                    2:{'name': 'Large optic cup', 'description': 'An enlarged optic cup may indicate glaucoma or other optic nerve conditions.', 'link': 'https://my.clevelandclinic.org/health/diseases/4212-glaucoma'},
                    3:{'name': 'Diabetic Retinopathy', 'description': 'Damage to the retina caused by complications of diabetes, which can eventually lead to blindness.', 'link': 'https://my.clevelandclinic.org/health/diseases/8591-diabetic-retinopathy'},
                    4:{'name': 'Possible glaucoma', 'description': 'A group eye conditions that damage the optic nerve, which is crucial for good vision', 'link': 'https://my.clevelandclinic.org/health/diseases/4212-glaucoma'},
                    5:{'name': 'Severe hypertensive retinopathy', 'description': 'Retinal damage associated with severe high blood pressure', 'link': 'https://my.clevelandclinic.org/health/diseases/25100-hypertensive-retinopathy'},
                    6:{'name': 'Disc Abnormalities', 'description': 'Abnormalities in the optic disc, which may indicate various ocular or systemic conditions.', 'link': 'https://pubmed.ncbi.nlm.nih.gov/21601065/#:~:text=Abnormalities%20of%20the%20optic%20disc%20may%20reflect%20eye%20disease%20(such,ganglion%20cells%20of%20one%20eye.'},
                    7:{'name': 'Dragged Disc', 'description': 'A condition where the optic disc appears displaced or dragged.', 'link': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4463564/#:~:text=Temporal%20dragging%20of%20the%20disc,pigmenti%2C%20and%20congenital%20retinal%20folds.'}, 
                    8:{'name': 'Retinitis pigmentosa', 'description': 'A group of genetic disorders that involve a breakdown and loss of cells in the retina.', 'link': 'https://my.clevelandclinic.org/health/diseases/17429-retinitis-pigmentosa'},
                    9:{'name': 'Bietti crystalline dystrophy', 'description': 'A rare genetic eye condition characterized by the presence of crystals in the retina.', 'link': 'https://eyewiki.aao.org/Bietti_Crystalline_Dystrophy#:~:text=Bietti\'s%20Crystalline%20Dystrophy%20(BCD)%2C,sclerosis%20of%20the%20choroidal%20vessels.'}, 
                    10:{'name': 'Peripheral retinal degeneration and break', 'description': 'Degeneration and tearing of the peripheral retina.', 'link': 'https://www.talleyeyeinstitute.com/peripheral-retina-breaks#:~:text=Peripheral%20Retina%20Breaks%2C%20tears%20or%20holes%3A&text=A%20retinal%20hole%20(atrophic%20hole,a%20condition%20called%20lattice%20degeneration.'},
                    11:{'name': 'Myelinated nerve fiber', 'description': 'An unusual appearance of the nerve fiber layer of the retina.', 'link': 'https://eyewiki.aao.org/Myelinated_Retinal_Nerve_Fiber_Layer'},
                    12:{'name': 'Vitreous particles', 'description': 'Particles in the vitreous humor of the eye, which can be a sign of age or various conditions.', 'link': 'https://my.clevelandclinic.org/health/symptoms/14209-eye-floaters-myodesopias'},
                    13:{'name': 'Fundus neoplasm', 'description': 'Tumors or growths in the back part of the eye.', 'link': 'https://en.wikipedia.org/wiki/Eye_neoplasm'},
                    14:{'name': 'BRVO', 'description': 'Branch Retinal Vein Occlusion, a blockage of the small veins in the retina.', 'link': 'https://eyewiki.aao.org/Branch_Retinal_Vein_Occlusion'}, 
                    15:{'name': 'CRVO', 'description': 'Central Retinal Vein Occlusion, a blockage of the main vein in the retina.', 'link': 'https://eyewiki.aao.org/Central_Retinal_Vein_Occlusion'}, 
                    16:{'name': 'Massive hard exudates', 'description': 'Large deposits of lipids and proteins in the retina.', 'link': 'https://www.vagelos.columbia.edu/departments-centers/ophthalmology/education/digital-reference-ophthalmology/vitreous-and-retina/retinal-vascular-diseases/hard-exudates'}, 
                    17:{'name': 'Yellow-white spots-flecks', 'description': 'Spots in the retina that can be a sign of various conditions.', 'link': 'https://kellogg.umich.edu/theeyeshaveit/opticfundus/tbl_yellow_white_things.html'}, 
                    18:{'name': 'Cotton-wool spots', 'description': 'Small, white, fluffy-looking patches on the retina.', 'link': 'https://eyewiki.aao.org/Cotton_Wool_Spots'}, 
                    19:{'name': 'Vessel tortuosity', 'description': 'Twisting or distortion of the blood vessels in the retina.', 'link': 'https://pubmed.ncbi.nlm.nih.gov/21146228/#:~:text=Retinal%20vascular%20tortuosity%20was%20defined,by%20the%20total%20path%20length.'}, 
                    20:{'name': 'Chorioretinal atrophy-coloboma', 'description': 'Thinning or absence of the retinal layers.', 'link': 'https://www.orpha.net/en/disease/detail/435930#:~:text=Colobomatous%20optic%20disc%2Dmacular%20atrophy%2Dchorioretinopathy%20syndrome,-Suggest%20an%20update&text=A%20rare%20genetic%20eye%20disease,optic%20disc)%20and%20macular%20atrophy.'}, 
                    21:{'name': 'Preretinal hemorrhage', 'description': 'Bleeding in front of the retina.', 'link': 'https://www.vagelos.columbia.edu/departments-centers/ophthalmology/education/digital-reference-ophthalmology/vitreous-and-retina/retinal-vascular-diseases/preretinal-hemorrhage'}, 
                    22:{'name': 'Fibrosis', 'description': 'Formation of excess fibrous connective tissue in the retina.', 'link': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8584003/#:~:text=Fibrosis%20of%20the%20cornea%20can,vision%20%5B6%2C7%5D.'}, 
                    23:{'name': 'Laser Spots', 'description': 'Marks on the retina from previous laser treatment.', 'link': 'https://medlineplus.gov/ency/article/007664.htm'}, 
                    24:{'name': 'Silicon oil in eye', 'description': 'Presence of silicone oil used in retinal surgery.', 'link': 'https://www.vrmny.com/procedures/silicone-oil/'}, 
                    25:{'name': 'Blur fundus without PDR', 'description': 'Blurred fundus without Proliferative Diabetic Retinopathy.', 'link': 'https://www.researchgate.net/figure/Examples-of-average-quality-fundus-images-a-Blur-b-Dark-c-Uneven-Illumination_fig2_340123815'}, 
                    26:{'name': 'Blur fundus with suspected PDR', 'description': 'Blurred fundus with suspected Proliferative Diabetic Retinopathy.', 'link': 'https://www.reviewofophthalmology.com/article/treating-proliferative-diabetic-retinopathy'}, 
                    27:{'name': 'RAO', 'description': 'Retinal Artery Occlusion, a blockage of the retinal artery.', 'link': 'https://www.aao.org/eye-health/diseases/what-is-stroke-affecting-eye'}, 
                    28:{'name': 'Rhegmatogenous RD', 'description': 'Rhegmatogenous Retinal Detachment, a type of retinal detachment.', 'link': 'https://www.mayoclinic.org/diseases-conditions/retinal-detachment/symptoms-causes/syc-20351344'}, 
                    29:{'name': 'CSCR', 'description': 'Central Serous Chorioretinopathy, a condition where fluid builds up under the retina.', 'link': 'https://www.asrs.org/patients/retinal-diseases/21/central-serous-chorioretinopathy#:~:text=Central%20serous%20chorioretinopathy%2C%20commonly%20referred,condition%20more%20commonly%20than%20women.'}, 
                    30:{'name': 'VKH disease', 'description': 'Vogt-Koyanagi-Harada disease, an autoimmune condition affecting the eyes.', 'link': 'https://rarediseases.org/rare-diseases/vogt-koyanagi-harada-disease/'}, 
                    31:{'name': 'Maculopathy', 'description': 'Any disease or pathological condition affecting the macula.', 'link': 'https://www.drugwatch.com/health/maculopathy/#:~:text=Maculopathy%2C%20also%20known%20as%20macular,loss%2C%20usually%20in%20both%20eyes.'}, 
                    32:{'name': 'ERM', 'description': 'Epiretinal Membrane, a thin layer of tissue on the retina.', 'link': 'https://www.asrs.org/patients/retinal-diseases/19/epiretinal-membranes'}, 
                    33:{'name': 'MH', 'description': 'Macular Hole, a small break in the macula.', 'link': 'https://my.clevelandclinic.org/health/diseases/14208-macular-hole'}, 
                    34:{'name': 'Pathological myopia', 'description': 'Severe myopia associated with degenerative changes in the eye.', 'link': 'https://eyewiki.org/Pathologic_Myopia_(Myopic_Degeneration)'}}

    # Streamlit UI
    st.title("Eye Condition Classifier")
    st.write("Upload a fundus image and if it isn't healthy, the model will classify it into one of 36 eye conditions.")

    # Image upload
    uploaded_file = st.file_uploader("Upload a fundus image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Fundus Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img_array = preprocess_image(img)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        disease = disease_info[predicted_class]
        # Display the result
        st.write(f"The predicted eye disease is: **{disease['name']}**")
        st.write(f"Description: {disease['description']}")
        st.markdown(f"[Resources related to {disease['name']}]({disease['link']})")
        st.write("Disclaimer: This tool is for reference purposes only and should not be used as a substitute for professional medical advice or diagnosis. Always consult a qualified healthcare professional for diagnosis and treatment options.")
