with open('readme.txt','r') as f_read:
    content = f_read.read()
    print(content)

with open('readme.txt','w') as f_write:
    f_write.write('This is Standard way to read txt file')
    f_write.write('after with open, write text file name and open mood')
    f_write.close()


with open('readme.txt','+w') as f_pwrite:
    f_pwrite.write('mood w, means write')
    f_pwrite.write('mood')
    f_pwrite.close()