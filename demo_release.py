import argparse
import matplotlib.pyplot as plt
from PIL import Image
import base64

from colorizers import *

# #######################image upload section
import os
from app import app
# import urllib.request
from flask import flash, request, redirect, url_for, render_template, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def color_img(filename):
#     pass
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0, no-cache, no-store'
    return response


# @app.route('/')
# def upload_form():
#     return render_template('upload.html')


@app.route('/api', methods=[ 'POST'])
def upload_image():
    if  False and 'file' not in request.files:
        print("Not in files>>>>>>>")
        flash('No file part')
        return redirect(request.url)
    print(request)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image.thumbnail((256,256))
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')
        # return render_template('upload.html', filename=filename)
        return jsonify({"message":"file uploaded"})
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        # return redirect(request.url)
        return jsonify({"message":"Allowed image types are -> png, jpg, jpeg, gif"}),400


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/api/color_output/<filename>')
def color_output(filename):
    print(filename)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='static/imgs/ansel_adams3.jpg')
    parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                        help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
    # opt = parser.parse_args()

    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    # colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    # if (opt.use_gpu):
    #     colorizer_eccv16.cuda()
    #     colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    # img = load_img(opt.img_path)
    print(filename)
    img = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print(filename)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    # if (opt.use_gpu):
    #     tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    # img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    # out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
    # plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)
    print('trying to save file...')
    plt.imsave('static/imgs_out/output.png', out_img_eccv16)
    
    with open('static/imgs_out/output.png', "rb") as image_file:
        # Read the binary data of the image file
        image_binary_data = image_file.read()
        
        # Encode the binary data to base64
        base64_encoded = base64.b64encode(image_binary_data)
        
        # Decode the base64 bytes to a string (optional)
        base64_string = base64_encoded.decode('utf-8')

    # plt.imsave('imgs_out/' +filename, out_img_siggraph17)

    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # plt.title('Original')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(img_bw)
    # plt.title('Input')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(out_img_eccv16)
    # plt.title('Output (ECCV 16)')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(out_img_siggraph17)
    # plt.title('Output (SIGGRAPH 17)')
    # plt.axis('off')
    # plt.show()

    # print('display_image filename: ' + filename)
    # return render_template('output.html', filename=filename)
    return jsonify({"data":base64_string})

@app.route('/', defaults={'path': ''})
@app.route("/<string:path>")
@app.route('/<path:path>')
def catch_all(path):

    # current_current_app.logger.info('path==', path)

    # return send_from_directory(current_app.static_folder, "index.html")
    print(current_app.static_folder+'/index.html')
    return current_app.send_static_file('index.html')
# #############################


if __name__ == "__main__":
    app.run(debug=True)
