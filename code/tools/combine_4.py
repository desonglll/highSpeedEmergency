import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def combine_four_images_with_pyplot(img1_path, img2_path, img3_path, img4_path,
                                    title_1, title_2, title_3, title_4,
                                    orientation='horizontal'):
    # 读取图片
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)
    img3 = mpimg.imread(img3_path)
    img4 = mpimg.imread(img4_path)

    # 创建图形和子图
    if orientation == 'horizontal':
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))  # 2行2列
        ax[0, 0].imshow(img1)
        ax[0, 0].axis('off')  # 不显示坐标轴
        ax[0, 0].set_title(title_1)

        ax[0, 1].imshow(img2)
        ax[0, 1].axis('off')  # 不显示坐标轴
        ax[0, 1].set_title(title_2)

        ax[1, 0].imshow(img3)
        ax[1, 0].axis('off')  # 不显示坐标轴
        ax[1, 0].set_title(title_3)

        ax[1, 1].imshow(img4)
        ax[1, 1].axis('off')  # 不显示坐标轴
        ax[1, 1].set_title(title_4)

    elif orientation == 'vertical':
        fig, ax = plt.subplots(4, 1, figsize=(6, 24))  # 4行1列
        ax[0].imshow(img1)
        ax[0].axis('off')  # 不显示坐标轴
        ax[0].set_title(title_1)

        ax[1].imshow(img2)
        ax[1].axis('off')  # 不显示坐标轴
        ax[1].set_title(title_2)

        ax[2].imshow(img3)
        ax[2].axis('off')  # 不显示坐标轴
        ax[2].set_title(title_3)

        ax[3].imshow(img4)
        ax[3].axis('off')  # 不显示坐标轴
        ax[3].set_title(title_4)

    # 调整子图间距
    plt.tight_layout()

    # 显示合并后的图片
    plt.show()
