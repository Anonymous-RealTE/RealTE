from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

path = '微软雅黑.ttf'
font = ImageFont.truetype(path, size=64, encoding="utf-8")
import lib2.config.my_alphabet as alphabets
alphabet = alphabets.alphabet
count = 0
for c_i in range(len(alphabet)):
    c = alphabet[c_i]
    chars_w, chars_h = font.getsize(c)
    random_glyph = Image.new('RGB', (chars_w, chars_h))
    draw_2 = ImageDraw.Draw(random_glyph)
    draw_2.text((0, 0), c, font=font, color=(255,0,255))
    glyph_processing = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5),
            ]
    )
    random_glyph = - glyph_processing(random_glyph)
    random_glyph = transforms.Resize((64, 64))(random_glyph)
    glyph_vis = Image.fromarray(((random_glyph / 2 + 0.5) * 255).detach().numpy().transpose(1,2,0).astype("uint8"))
    glyph_vis.save("utils/alphabet_img/{}.png".format(c_i))

    count += 1
    if count % 1000 == 0:
        print(count)