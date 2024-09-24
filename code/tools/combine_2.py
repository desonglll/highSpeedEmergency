import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def combine_images_with_pyplot(img1_path, img2_path, title_1, title_2, orientation='horizontal'):
    # 读取图片
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)

    # 创建图形和子图
    if orientation == 'horizontal':
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1行2列
        ax[0].imshow(img1)
        ax[0].axis('off')  # 不显示坐标轴
        ax[0].set_title(title_1)

        ax[1].imshow(img2)
        ax[1].axis('off')  # 不显示坐标轴
        ax[1].set_title(title_2)

    elif orientation == 'vertical':
        fig, ax = plt.subplots(2, 1, figsize=(6, 12))  # 2行1列
        ax[0].imshow(img1)
        ax[0].axis('off')  # 不显示坐标轴
        ax[0].set_title(title_1)

        ax[1].imshow(img2)
        ax[1].axis('off')  # 不显示坐标轴
        ax[1].set_title(title_2)

    # 调整子图间距
    plt.tight_layout()

    # 显示合并后的图片
    plt.show()
