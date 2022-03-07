from tqdm import tqdm
import zipfile
import os
import shutil
from glob import glob

#zip file들을 extract해주는 코드
def zip_file_extract(file_list,result_dir):
    for file_name in tqdm(file_list):
        with zipfile.ZipFile(file_name, 'r') as zip:
            zipInfo = zip.infolist()
            for member in zipInfo:
                member.filename = member.filename.encode("cp437").decode("cp949")
                zip.extract(member,result_dir)


# [ex) 꼬깔콘, 오징어땅콩] 과 같은 여러 폴더에 있는 내용을 하나의 폴더로 다 모아주는 코드 (출처 : https://gagadi.tistory.com/9)
def read_all_file(path):
    output = os.listdir(path) #directory 내에 모든 파일 or directory 리스트 출력
    file_list = []
    for i in output:
        if os.path.isdir(path+"/"+i):
            file_list.extend(read_all_file(path+"/"+i))
        elif os.path.isfile(path+"/"+i):
            file_list.append(path+"/"+i)
    return file_list

def copy_all_file(file_list, new_path):

    # shutil.copy의 speed를 증가시키는 코드
    def _copyfileobj_patched(fsrc, fdst, length=16 * 1024 * 1024):
        """Patches shutil copyfileobj method to hugely improve copy speed"""
        while 1:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)

    shutil.copyfileobj = _copyfileobj_patched

    for src_path in file_list:
        file = src_path.split("/")[-1]
        shutil.copyfile(src_path, new_path + "/" + file)
        print("파일 {} 작업 완료".format(file)) # 작업한 파일명 출력

def copy_img_label(img_list,label_list,result_folder_path):
    # shutil.copy의 speed를 증가시키는 코드
    def _copyfileobj_patched(fsrc, fdst, length=16 * 1024 * 1024):
        """Patches shutil copyfileobj method to hugely improve copy speed"""
        while 1:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)

    shutil.copyfileobj = _copyfileobj_patched
    for i in tqdm(img_list):
         if os.path.exists(i):
            shutil.copy(i,result_folder_path)
    for i in tqdm(label_list):
        if os.path.isfile(i):
            if not "meta" in i:
                shutil.copy(i,result_folder_path)