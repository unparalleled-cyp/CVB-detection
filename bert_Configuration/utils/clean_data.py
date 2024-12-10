# -*- coding: utf-8 -*-
"""
@File: clean_data.py 
@Time : 2022/5/5 21:21

@dec: 将字符进行格式转换
"""

import re


def dbc2sbc(ustring):
    """
        全角转半角
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not 0x0021 <= inside_code <= 0x7e:
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return " ".join(rstring.split(" "))


def clean_string(s):
    """
    清洗字符串  保留tab和换行符
    但是不同人的预处理方式可能会有一定差别，所以这个模块可能需要换一下
    """
    s = dbc2sbc(s)
    # s = re.sub(r'[\xa0]', ' ', s)
    # s = s.replace("\\t", " ").replace("\\", " ")
    # s = re.sub(r'\[emotion', '[', s)
    # s = re.sub(r'emotion\]', ']', s)
    #s = re.sub(r'\。', '.', s)
    #s = re.sub(r'\，', ',', s)
    #s = re.sub(r'\！', '!', s)
    #s = re.sub(r'\？', '?', s)
    #s = re.sub(r'\：', ':', s)
    #s = re.sub(r'\（', '(', s)
    #s = re.sub(r'\）', ')', s)
    #s = re.sub(r'\】', ']', s)
    #s = re.sub(r'\【', '[', s)
    #s = re.sub(r'\~', '~', s)
    #s = re.sub(r'\—', '-', s)
    #s = re.sub(r'\；', ';', s)
    #s = re.sub(r'\‘', "'", s)
    #s = re.sub(r'\“', '"', s)
    s = s.replace("\u0007", " ")
    # s = re.sub(r"[\xa0\n\t\ ]+", " ", s)
    s = re.sub(r"\xa0", " ", s)  # re.sub(r"[\xa0\s]+", " ", s)
    s = re.sub(r"[ ]+", " ", s)  # 去掉连续空格
    s = re.sub(r"[\n]+", "\n", s)  # 去掉连续换行符
    s = re.sub(r"[\t]+", "\t", s)  # 去掉连续tab符
    s = re.sub(r"[ ]*\n[ ]*", "\\n", s)  # 去掉换行符前后的空格
    s = re.sub(r"[\t]*\n[\t]*", "\\n", s)  # 去掉换行符前后的tab符
    s = re.sub(r"[ ]*\t[ ]*", "\\t", s)  # 去掉tab符前后的空格
    # s = re.sub("\n", " ", s)
    # s = re.sub("[ ]+", " ", s)
    # s = s.upper()
    return s

def cut_sentence(s):
    s = re.sub(r'((\[emotion.*?emotion\])|(.))', r"\1 ", s)
    s = s.replace('emotion]',']').replace('[emotion','[')
    s = re.sub(r"[ ]+", " ", s)  # 去掉连续空格
    return s.split()

if __name__ == "__main__":
    s = "\x03\x00\x00\x13\x0eà\x00\x00\x00\x00\x00\x01\x00\x08\x00\x02\x00\x00\x00"
    print(s)

    s = '''[emotiondogeemotion]   【    #为赚钱制售假新冠疫苗约5.8万支#[emotion怒emotion]犯罪嫌疑人已被批捕】#最高检发布涉新冠疫苗犯罪典型案例#：孔某、乔某购买预灌封注射器，用生理盐水制造假新冠疫苗，后期因生理盐水不足，还以矿泉水代替。之后对外伪称是“从内部渠道拿到的正品新冠疫苗”，以致假疫苗流入社会。初步查明，孔某、乔某等人制造并销售假新冠疫苗约5.8万支，获利约1800万元。2020年12月25日，检察机关决定对犯罪嫌疑人孔某、乔某等人批准逮捕。（总台央视记者程琴）
😡😡😡诸神愤怒⚡'''
    # print(clean_string(s))
    print(cut_sentence(clean_string(s)))