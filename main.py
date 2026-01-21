import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import os
import webcolors
import random

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- CSS color map (unchanged) ---
CSS3_COLORS = {
    "aliceblue": "#f0f8ff", "antiquewhite": "#faebd7", "aqua": "#00ffff",
    "aquamarine": "#7fffd4", "azure": "#f0ffff", "beige": "#f5f5dc",
    "bisque": "#ffe4c4", "black": "#000000", "blanchedalmond": "#ffebcd",
    "blue": "#0000ff", "blueviolet": "#8a2be2", "brown": "#a52a2a",
    "burlywood": "#deb887", "cadetblue": "#5f9ea0", "chartreuse": "#7fff00",
    "chocolate": "#d2691e", "coral": "#ff7f50", "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc", "crimson": "#dc143c", "cyan": "#00ffff",
    "darkblue": "#00008b", "darkcyan": "#008b8b", "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9", "darkgreen": "#006400", "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b", "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00", "darkorchid": "#9932cc", "darkred": "#8b0000",
    "darksalmon": "#e9967a", "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b", "darkslategray": "#2f4f4f",
    "darkturquoise": "#00ced1", "darkviolet": "#9400d3",
    "deeppink": "#ff1493", "deepskyblue": "#00bfff", "dimgray": "#696969",
    "dodgerblue": "#1e90ff", "firebrick": "#b22222", "floralwhite": "#fffaf0",
    "forestgreen": "#228b22", "fuchsia": "#ff00ff", "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff", "gold": "#ffd700", "goldenrod": "#daa520",
    "gray": "#808080", "green": "#008000", "greenyellow": "#adff2f",
    "honeydew": "#f0fff0", "hotpink": "#ff69b4", "indianred": "#cd5c5c",
    "indigo": "#4b0082", "ivory": "#fffff0", "khaki": "#f0e68c",
    "lavender": "#e6e6fa", "lavenderblush": "#fff0f5", "lawngreen": "#7cfc00",
    "lemonchiffon": "#fffacd", "lightblue": "#add8e6",
    "lightcoral": "#f08080", "lightcyan": "#e0ffff",
    "lightgoldenrodyellow": "#fafad2", "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90", "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a", "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa", "lightslategray": "#778899",
    "lightsteelblue": "#b0c4de", "lightyellow": "#ffffe0",
    "lime": "#00ff00", "limegreen": "#32cd32", "linen": "#faf0e6",
    "magenta": "#ff00ff", "maroon": "#800000", "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd", "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db", "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee", "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc", "mediumvioletred": "#c71585",
    "midnightblue": "#191970", "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1", "moccasin": "#ffe4b5", "navajowhite": "#ffdead",
    "navy": "#000080", "oldlace": "#fdf5e6", "olive": "#808000",
    "olivedrab": "#6b8e23", "orange": "#ffa500", "orangered": "#ff4500",
    "orchid": "#da70d6", "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98", "paleturquoise": "#afeeee",
    "palevioletred": "#db7093", "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9", "peru": "#cd853f", "pink": "#ffc0cb",
    "plum": "#dda0dd", "powderblue": "#b0e0e6", "purple": "#800080",
    "red": "#ff0000", "rosybrown": "#bc8f8f", "royalblue": "#4169e1",
    "saddlebrown": "#8b4513", "salmon": "#fa8072", "sandybrown": "#f4a460",
    "seagreen": "#2e8b57", "seashell": "#fff5ee", "sienna": "#a0522d",
    "silver": "#c0c0c0", "skyblue": "#87ceeb", "slateblue": "#6a5acd",
    "slategray": "#708090", "snow": "#fffafa", "springgreen": "#00ff7f",
    "steelblue": "#4682b4", "tan": "#d2b48c", "teal": "#008080",
    "thistle": "#d8bfd8", "tomato": "#ff6347", "turquoise": "#40e0d0",
    "violet": "#ee82ee", "wheat": "#f5deb3", "white": "#ffffff",
    "whitesmoke": "#f5f5f5", "yellow": "#ffff00", "yellowgreen": "#9acd32"
}

@app.route("/")
def index():
    sample_colors = random.sample(list(CSS3_COLORS.items()), 30)
    return render_template("index.html", sample_colors=sample_colors)


def extract_colors(image_path, num_colors=10):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)

    kmeans = MiniBatchKMeans(n_clusters=num_colors, batch_size=2048, n_init="auto")
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    hex_codes = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
    return [(hex_code, closest_color_name(hex_code)) for hex_code in hex_codes]


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]
    if file.filename == "":
        return "No selected file"

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_path)

    hex_codes = extract_colors(image_path)
    return render_template("upload.html", hex_codes=hex_codes, filename=filename)


def closest_color_name(hex_code):
    r1, g1, b1 = webcolors.hex_to_rgb(hex_code)

    min_diff = float("inf")
    closest_name = None

    for name, hex_value in CSS3_COLORS.items():
        r2, g2, b2 = webcolors.hex_to_rgb(hex_value)
        diff = (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2

        if diff < min_diff:
            min_diff = diff
            closest_name = name

    return closest_name


if __name__ == "__main__":
    app.run(debug=True)
