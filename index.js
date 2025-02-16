import { Jimp, intToRGBA, diff } from "https://cdn.jsdelivr.net/npm/jimp@1.6.0/dist/browser/index.min.js";

//import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";

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

const model = await load_model();
console.log(model);

async function remove_bg(img_buffer) {
    const session = await ort.InferenceSession.create(model);
    const dims = [1, 3, 1024, 1024];
    const norm_mean = 0.5;
    const norm_std = 1.0;

    var image_height = 0;
    var image_width = 0;

    var red_arr = [];
    var green_arr = [];
    var blue_arr = [];

    const image = await Jimp.fromBuffer(img_buffer);
    
    // save original dimensions
    image_height = image.bitmap.height; 
    image_width = image.bitmap.width;

    image.resize({w: dims[2], h: dims[3]});
   
    // re-arrange pixels 
    for (let i = 0; i < dims[2]; i++) {
        for (let j = 0; j < dims[3]; j++) {
            const rgba = intToRGBA(image.getPixelColor(j, i));
            red_arr.push(((rgba.r / 255.0) - norm_mean) / norm_std);
            green_arr.push(((rgba.g / 255.0) - norm_mean) / norm_std);
            blue_arr.push(((rgba.b / 255.0) - norm_mean) / norm_std);
        }
    }
    const img_arr = red_arr.concat(green_arr).concat(blue_arr);

    const img_tensor = new ort.Tensor("float32", img_arr, dims);

    console.log("running inference");
    const feeds = {};
    feeds[session.inputNames[0]] = img_tensor;
    const model_output = await session.run(feeds);
    const output = model_output[session.outputNames[0]];
    console.log("inference complete");

    var max_value = 0;
    var min_value = Infinity;
    for (let i = 0; i < output.size; i++) {
        if (output.data[i] > max_value) {
            max_value = output.data[i];
        }
        if (output.data[i] < min_value) {
            min_value = output.data[i];
        }
    }
    
    const mask_buffer = new Uint8Array(output.size * 4);
    for (let i = 0; i < output.size; i++) {
        const output_norm = 255 * ((output.data[i] - min_value)  / (max_value - min_value));
        
        var pixel_value = 0;

        // tuneable, 170 is arbitrary value 
        if (output_norm > 170) {
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

    var preview_image = image.mask(image_mask);
    preview_image.resize({w: image_width, h: image_height});

    image_mask.resize({w: image_width, h: image_height});
    var mask = document.querySelector("#mask_image");
    mask.src = await image_mask.getBase64("image/png");

    var preview = document.querySelector("#preview_image");
    preview.src = await preview_image.getBase64("image/png");
}

const upload_element = document.querySelector("#upload");

upload_element.addEventListener("change", function() {
    if (this.files && this.files[0]) {
        var img = document.querySelector("#original_image");
        // no clue what memory is being saved here  
        img.onload = () => {
            URL.revokeObjectURL(img.src);  // no longer needed, free memory
        }
        
        const img_url = URL.createObjectURL(this.files[0]); // set src to blob url
        img.src = img_url;
        console.log(img.src);

        this.files[0].arrayBuffer().then(remove_bg);
    }
});
