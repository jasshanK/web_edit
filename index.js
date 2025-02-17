import { Jimp, intToRGBA, diff } from "https://cdn.jsdelivr.net/npm/jimp@1.6.0/dist/browser/index.min.js";

var model;
var mask_cutoff = 170;
var inference_complete = false;
const dims = [1, 3, 1024, 1024]; //model_dims
var image_height = 0;
var image_width = 0;
var output;
var max_value = 0;
var min_value = Infinity;

const load_model_btn = document.querySelector("#load_model_btn");
const mask_range_slider = document.querySelector("#mask_range");
const upload_element = document.querySelector("#upload");

async function load_model() {
    var model_array = [];
    for (let i = 0; i < 4; i++) {
        const response = await fetch(`./model/shard${i}`);
        const result = await response.arrayBuffer();
        model_array[i] = result;
    }

    var model_full = new Uint8Array(
        model_array[0].byteLength +
        model_array[1].byteLength +
        model_array[2].byteLength +
        model_array[3].byteLength);

    model_full.set(new Uint8Array(model_array[0]), 0);
    model_full.set(new Uint8Array(model_array[1]), model_array[0].byteLength);
    model_full.set(new Uint8Array(model_array[2]), model_array[0].byteLength + model_array[1].byteLength);
    model_full.set(new Uint8Array(model_array[3]), model_array[0].byteLength + model_array[1].byteLength + model_array[2].byteLength);

    return model_full;
}

async function display_res(original_image) {
    const mask_buffer = new Uint8Array(output.size * 4);
    for (let i = 0; i < output.size; i++) {
        const output_norm = 255 * ((output.data[i] - min_value)  / (max_value - min_value));
        
        var pixel_value = 0;

        if (output_norm > mask_cutoff) {
            pixel_value = 255;
        }

        mask_buffer[4*i+0] = pixel_value;
        mask_buffer[4*i+1] = pixel_value;
        mask_buffer[4*i+2] = pixel_value;
        mask_buffer[4*i+3] = 255;
    }

    let image_mask = new Jimp({ data: mask_buffer, width: dims[2], height: dims[3] }, 
        (err, _image) => {
            if (err) throw err;
        });
        
    const preview_image = original_image.mask(image_mask);

    image_mask.resize({w: image_width, h: image_height});
    const mask = document.querySelector("#mask_image");
    mask.src = await image_mask.getBase64("image/png");

    preview_image.resize({w: image_width, h: image_height});
    const preview = document.querySelector("#preview_image");
    preview.src = await preview_image.getBase64("image/png");
}

async function remove_bg(img_buffer) {
    const session = await ort.InferenceSession.create(model);
    const norm_mean = 0.5;
    const norm_std = 1.0;

    var red_arr = [];
    var green_arr = [];
    var blue_arr = [];

    const upload_image = await Jimp.fromBuffer(img_buffer);
    
    // save original dimensions
    image_height = upload_image.bitmap.height; 
    image_width = upload_image.bitmap.width;

    upload_image.resize({w: dims[2], h: dims[3]});
   
    // re-arrange pixels 
    for (let i = 0; i < dims[2]; i++) {
        for (let j = 0; j < dims[3]; j++) {
            const rgba = intToRGBA(upload_image.getPixelColor(j, i));
            red_arr.push(((rgba.r / 255.0) - norm_mean) / norm_std);
            green_arr.push(((rgba.g / 255.0) - norm_mean) / norm_std);
            blue_arr.push(((rgba.b / 255.0) - norm_mean) / norm_std);
        }
    }
    const img_arr = red_arr.concat(green_arr).concat(blue_arr);

    const img_tensor = new ort.Tensor("float32", img_arr, dims);

    alert("running inference");
    const feeds = {};
    feeds[session.inputNames[0]] = img_tensor;
    const model_output = await session.run(feeds);
    output = model_output[session.outputNames[0]];
    alert("inference complete");
    inference_complete = true;

    for (let i = 0; i < output.size; i++) {
        if (output.data[i] > max_value) {
            max_value = output.data[i];
        }
        if (output.data[i] < min_value) {
            min_value = output.data[i];
        }
    }

    display_res(upload_image);
}

load_model_btn.onclick = async () => {
    model = await load_model();
    alert("Model loaded");
};

upload_element.addEventListener("change", function() {
    if (this.files && this.files[0]) {
        var img = document.querySelector("#original_image");
        
        const img_url = URL.createObjectURL(this.files[0]); // set src to blob url
        img.src = img_url;
        
        this.files[0].arrayBuffer().then(remove_bg);
    }
});

mask_range_slider.addEventListener("input", async () => {
    mask_cutoff = mask_range_slider.value;
    
    if (inference_complete) {
        upload_element.files[0].arrayBuffer().then( async (img_buffer) => {
            const upload_image = await Jimp.fromBuffer(img_buffer);
            upload_image.resize({w: dims[2], h: dims[3]});
            display_res(upload_image);
        });
    }
});
