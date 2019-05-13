import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import os

font_file = 'DroidSansFallbackFull.ttf'
my_font = FontProperties(fname=font_file)


def visualize_attention_per_head(tmp_Vocab, tower_mems_id_np, attn_prob, next_word_index, file_path, length):
    # todo 作图
    head_num = 10
    xLabel = tmp_Vocab.get_symbols([tower_mems_id_np[0][0][i][0] for i in range(-100, 0, 1)])
    yLabel = list(range(0, 16))

    datas = []
    for l in range(head_num):  #head
        data = []
        for i in range(len(yLabel)):
            temp = []
            for j in range(len(xLabel)):
                k = attn_prob[0][i][0][-100+j][0][l]
                temp.append(k)
            data.append(temp)
        datas.append(data)

    # 作图阶段
    fig = plt.figure(figsize=(48, 48))

    for k in range(head_num):
        ax = fig.add_subplot(5, 2, k+1)
        # plt.subplots_adjust(left=(k%2)*0.5+0.05, right=(k%2)*0.5+0.45)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # 定义横纵坐标的刻度
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel, fontproperties=my_font, rotation=45, size=5)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(datas[k], cmap=plt.cm.hot_r)
        plt.title(tmp_Vocab.get_sym(next_word_index)+' head:{}'.format(k), fontproperties=my_font)

    # 增加右侧的颜色刻度条
    # plt.colorbar(im)
    # 增加标题
    # file_path = 'tmp_img'
    if (os.path.exists(file_path) == False):
        os.makedirs(file_path)
    plt.savefig('{}/{}.png'.format(file_path, str(length)+tmp_Vocab.get_sym(next_word_index)))
    # # # show
    # plt.show()


def visualize_attention_per_layer(tmp_Vocab, tower_mems_id_np, attn_prob, next_word_index, file_path, length):
    # todo 作图
    head_num = 10
    layer_num = 16
    xLabel = tmp_Vocab.get_symbols([tower_mems_id_np[0][0][i][0] for i in range(100)])
    yLabel = list(range(0, 10))

    datas = []
    for i in range(layer_num):
        data = []
        for l in range(head_num):  # head
            temp = []
            for j in range(len(xLabel)):
                k = attn_prob[0][i][0][-100+j][0][l]
                temp.append(k)
            data.append(temp)
        datas.append(data)

    # 作图阶段
    fig = plt.figure(figsize=(48, 36))

    for k in range(layer_num):
        ax = fig.add_subplot(8, 2, k+1)
        # plt.subplots_adjust(left=(k%2)*0.5+0.05, right=(k%2)*0.5+0.45)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # 定义横纵坐标的刻度
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel, fontproperties=my_font, rotation=45, size=5)
        # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(datas[k], cmap=plt.cm.hot_r)
        plt.title(tmp_Vocab.get_sym(next_word_index)+' layer:{}'.format(k), fontproperties=my_font)

    if (os.path.exists(file_path) == False):
        os.makedirs(file_path)
    plt.savefig('{}/{}.png'.format(file_path, str(length)+tmp_Vocab.get_sym(next_word_index)))
    # # # show
    # plt.show()


def visualize_prob(tmp_Vocab, tmp_list, file_path, length):
    plt.figure()
    index_list = sorted(range(len(tmp_list)), key=lambda k: tmp_list[k], reverse=True)[:10]
    xlist = tmp_Vocab.get_symbols(index_list)
    ylist = []
    for i in range(len(index_list)):
        ylist.append(tmp_list[index_list[i]])
    plt.bar(range(len(xlist)), ylist)
    plt.xticks(range(len(xlist)), xlist, fontproperties=my_font, rotation=0, size=5)

    if (os.path.exists(file_path) == False):
        os.makedirs(file_path)
    plt.savefig('{}.png'.format(os.path.join(file_path, str(length))))
