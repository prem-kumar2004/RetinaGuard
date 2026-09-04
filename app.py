import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, render_template, request


APP_ROOT = Path(__file__).resolve().parent

MODEL_PATH = APP_ROOT / 'model' / 'best_model.keras'

IMAGE_SIZE = 224
CROP_THRESHOLD = 10
CROP_PADDING = 10
MAX_FILE_SIZE = 10 * 1024 * 1024

CLASS_NAMES = [
    'No DR',
    'Mild',
    'Moderate',
    'Severe',
    'Proliferative',
]

ALLOWED_EXTENSIONS = {
    '.png',
    '.jpg',
    '.jpeg',
}

TARGET_LAYER = 'top_conv'


app = Flask(
    __name__,
    template_folder=str(APP_ROOT / 'templates'),
)

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)


def preprocess_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32,
    )

    gray = tf.reduce_mean(image, axis=-1)

    mask = gray > tf.cast(
        CROP_THRESHOLD / 255.0,
        tf.float32,
    )

    coords = tf.where(mask)

    def crop_image():

        y_min = tf.cast(
            tf.reduce_min(coords[:, 0]),
            tf.int32,
        )

        y_max = tf.cast(
            tf.reduce_max(coords[:, 0]),
            tf.int32,
        )

        x_min = tf.cast(
            tf.reduce_min(coords[:, 1]),
            tf.int32,
        )

        x_max = tf.cast(
            tf.reduce_max(coords[:, 1]),
            tf.int32,
        )

        h = tf.cast(
            tf.shape(image)[0],
            tf.int32,
        )

        w = tf.cast(
            tf.shape(image)[1],
            tf.int32,
        )

        y_min = tf.maximum(
            y_min - CROP_PADDING,
            0,
        )

        x_min = tf.maximum(
            x_min - CROP_PADDING,
            0,
        )

        y_max = tf.minimum(
            y_max + CROP_PADDING + 1,
            h,
        )

        x_max = tf.minimum(
            x_max + CROP_PADDING + 1,
            w,
        )

        return image[y_min:y_max, x_min:x_max]

    image = tf.cond(
        tf.size(coords) > 0,
        crop_image,
        lambda: image,
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method='bilinear',
    )

    image = tf.clip_by_value(
        image,
        0.0,
        1.0,
    )

    return image


def assess_quality(image_path):

    image = tf.io.decode_image(
        tf.io.read_file(str(image_path)),
        channels=3,
        expand_animations=False,
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32,
    )

    arr = image.numpy()

    h, w = arr.shape[:2]

    gray = np.mean(arr, axis=2)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    black_fraction = float(np.mean(gray < 0.04))

    non_black = gray >= 0.04

    if np.any(non_black):

        ys, xs = np.where(non_black)

        y_range = float(ys.max() - ys.min() + 1) / max(h, 1)
        x_range = float(xs.max() - xs.min() + 1) / max(w, 1)

        fov_fraction = float(y_range * x_range)

    else:

        fov_fraction = 0.0

    flags = []

    if h < 100 or w < 100:
        flags.append('very_small_image')

    if brightness < 0.05:
        flags.append('very_dark')

    if brightness > 0.95:
        flags.append('very_bright')

    if contrast < 0.05:
        flags.append('very_low_contrast')

    if black_fraction > 0.85:
        flags.append('extreme_black_background')

    elif black_fraction > 0.70:
        flags.append('large_black_background')

    if fov_fraction < 0.15:
        flags.append('small_visible_fov')

    score = 1.0

    penalties = {
        'very_small_image': 0.40,
        'very_dark': 0.15,
        'very_bright': 0.15,
        'very_low_contrast': 0.20,
        'extreme_black_background': 0.15,
        'large_black_background': 0.08,
        'small_visible_fov': 0.15,
    }

    for flag in flags:
        score -= penalties.get(flag, 0.0)

    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.70:
        status = 'acceptable'
    elif score >= 0.40:
        status = 'review'
    else:
        status = 'ungradable'

    return {
        'status': status,
        'acceptable': status == 'acceptable',
        'score': score,
        'brightness': brightness,
        'contrast': contrast,
        'black_background_fraction': black_fraction,
        'fov_fraction': fov_fraction,
        'width': int(w),
        'height': int(h),
        'flags': flags,
        'advisory_only': True,
    }


def predict_image(image_path):
    """
    Run the frozen champion model on an already-resolved image path.

    The saved champion model already contains a final
    Dense(5, activation="softmax") layer, so the model output
    is already a probability vector.
    """

    image = preprocess_image(image_path)

    tensor = tf.expand_dims(
        image,
        axis=0,
    )

    # IMPORTANT:
    # The model already returns probabilities.
    # Do NOT apply another softmax here.
    probabilities = model(
        tensor,
        training=False,
    ).numpy()[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.shape != (5,):
        raise ValueError(
            f"Expected five probabilities, "
            f"got shape {probabilities.shape}"
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError(
            "Invalid model probabilities."
        )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Model probabilities do not sum to one."
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        'class_index': predicted_index,
        'grade': CLASS_NAMES[predicted_index],
        'confidence': confidence,
        'probabilities': [
            float(x)
            for x in probabilities
        ],
    }


def home():
    return render_template('index.html')



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'RetinaGuard',
        'model': 'B0_CLASSWEIGHTED_FINETUNED_224',
        'input_size': '224x224',
        'classes': 5,
    })


@app.route('/analyze', methods=['POST'])
def analyze():

    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image uploaded.',
        }), 400

    uploaded = request.files['image']

    filename = (
        uploaded.filename or ''
    ).strip()

    if not filename:
        return jsonify({
            'success': False,
            'error': 'Empty filename.',
        }), 400

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            'success': False,
            'error': 'Unsupported image format. Use PNG or JPEG.',
        }), 400

    temp_path = None

    try:

        with NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:

            temp_path = Path(temp_file.name)
            uploaded.save(str(temp_path))

        quality = assess_quality(temp_path)

        prediction = predict_image(temp_path)

        return jsonify({
            'success': True,
            'model': {
                'name': 'B0_CLASSWEIGHTED_FINETUNED_224',
                'architecture': 'EfficientNetB0',
                'input_size': '224x224',
                'classes': CLASS_NAMES,
                'frozen': True,
            },
            'prediction': prediction,
            'quality': quality,
            'gradcam': {
                'available': True,
                'target_layer': TARGET_LAYER,
                'method': 'Grad-CAM',
                'note': 'Model explainability metadata. Not clinical evidence.',
            },
            'preprocessing': {
                'rgb': True,
                'crop_threshold': CROP_THRESHOLD,
                'crop_padding': CROP_PADDING,
                'resize': '224x224',
                'normalization': '[0,1]',
                'augmentation': False,
            },
            'safety': {
                'screening_support_only': True,
                'not_a_medical_diagnosis': True,
                'requires_healthcare_professional_review': True,
            },
        })

    except Exception as exc:

        return jsonify({
            'success': False,
            'error': str(exc),
        }), 500

    finally:

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False,
    )
