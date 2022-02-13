import json
LUNA = False
LUCID = True
path_lucid_score_train = r"C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\DanielNet_score_Lucid\danielNet_score_from_lucid.json"

# danielNet_score_from_lucid.json
lucid = json.load(open(path_lucid_score_train))
lucid = eval(lucid)

path_luna_score_train = r"C:\Repositories\luna\test_score"
# all_score_layer_conv2d_1_channel_2.json

path_luna_inference = r"C:\Repositories\luna\danielNet_score_from_inference_luna_2.json"
luna_interface = json.load(open(path_luna_inference))
luna_interface = eval(luna_interface)

path_lucid_inference = r"C:\Repositories\luna\danielNet_score_from_inference_lucid_2.json"
lucid_interface = json.load(open(path_lucid_inference))
lucid_interface = eval(lucid_interface)

danielNet_layers_luna  = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"]
danielNet_layers_lucid = ["Conv2D", "Conv2D_1", "Conv2D_2", "Conv2D_3", "Conv2D_4"]
danielNet_channels = [8, 16, 16, 32, 32]

for layer_name_luna, layer_name_lucid, num_of_channel in zip(danielNet_layers_luna, danielNet_layers_lucid, danielNet_channels):
    for channel_num in range(num_of_channel):
        if LUCID:
            lucid_score = lucid["{}_{}".format(layer_name_lucid, channel_num)]
            lucid_interface_score = lucid_interface["{}_{}".format(layer_name_luna, channel_num)]
            print(f"At layer {layer_name_luna}, channel {channel_num}, lucid score is {(-1) * lucid_score} and lucid interface score is {lucid_interface_score}")

        if LUNA:
            luna = json.load(open(f"{path_luna_score_train}/all_score_layer_{layer_name_luna}_channel_{channel_num}.json"))
            luna = eval(luna)
            luna_score = luna["loss"]

            luna_interface_score = luna_interface["{}_{}".format(layer_name_luna, channel_num)]
            print(f"At layer {layer_name_luna}, channel {channel_num}, luna score is {luna_score} and luna interface score is {luna_interface_score}")