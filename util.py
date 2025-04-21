import tensorflow as tf
import numpy as np

model = None
output_class = ["Batteries", "Clothes", "E-waste", "Glass", "Light Blubs", "Metal", "Organic", "Paper", "Plastic"]
data = {
    "Batteries": [
        "1. Sort by type (alkaline, lithium-ion, lead-acid).",
        "2. Tape terminals (alkaline) or store lithium-ion safely.",
        "3. Drop off at retailers (Best Buy, Home Depot) or recycling centers.",
        "4. Mail-in programs for hard-to-recycle batteries (Call2Recycle).",
        "Medium"
    ],
    "Clothes": [
        "1. Sort clothes by material (cotton, denim, polyester).",
        "2. Repurpose: Cut T-shirts into rags or braid denim into rugs.",
        "3. Donate wearable items to thrift stores or textile recycling bins.",
        "4. For industrial recycling, send to facilities that shred fabric into insulation.",
        "Easy"
    ],
    "E-waste": [
        "1. Wipe data from devices (phones, laptops).",
        "2. Remove batteries (recycle separately).",
        "3. Take to e-waste centers or retailer take-back programs.",
        "4. For DIY: Salvage circuit boards for art or copper from wires.",
        "Medium"
    ],
    "Glass": [
        "1. Rinse glass jars/bottles to remove residue.",
        "2. Sort by color (clear, green, brown).",
        "3. Crush into cullet or reuse as containers/planters.",
        "4. Deliver to glass recycling facilities (no ceramics mixed in).",
        "Easy"
    ],
    "Light Bulbs": [
        "1. Handle CFLs/LEDs carefully to avoid breakage (wear gloves).",
        "2. Store in original packaging for safe transport.",
        "3. Drop off at hardware stores or hazardous waste centers.",
        "4. For LEDs: Extract copper from wires if DIYing.",
        "Medium"
    ],
    "Metal": [
        "1. Separate ferrous (magnetic) and non-ferrous metals.",
        "2. Clean cans/bottles to remove food residue.",
        "3. Flatten to save space before recycling.",
        "4. Sell scrap metal or use cans for DIY planters.",
        "Easy"
    ],
    "Organic": [
        "1. Compost food scraps (no meat/dairy) with yard waste.",
        "2. Layer greens (food) and browns (leaves/paper) in a bin.",
        "3. Turn the pile weekly to aerate.",
        "4. Use finished compost in gardens.",
        "Easy"
    ],
    "Paper": [
        "1. Remove non-paper items (staples, plastic windows).",
        "2. Sort by grade (cardboard, office paper, newspapers).",
        "3. Shred for pet bedding or recycle via curbside bins.",
        "4. For crafts: Make seed paper or papier-mâché.",
        "Easy"
    ],
    "Plastic": [
        "1. Check resin codes (#1 PET, #2 HDPE are widely recyclable).",
        "2. Clean and remove labels/caps (recycle separately).",
        "3. Flatten bottles to save space.",
        "4. Avoid flexible plastics (bags/wraps) unless specified.",
        "Medium"
    ]
}


def load_artifacts():
    global model
    model = tf.keras.models.load_model("classifyWaste.h5")

def classify_waste(image_path):
	global model, output_class
	test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis = 0)
	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]
	return predicted_value, data[predicted_value][0], data[predicted_value][1], data[predicted_value][2],data[predicted_value][3],data[predicted_value][4]

