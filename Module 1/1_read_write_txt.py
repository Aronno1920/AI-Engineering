#read write text file

####General Way
#Read file
f_read = open('readme.txt','r')
content = f_read.read()
print(content)

f_read.close()

#write file
f_write = open('readme.txt','w')
f_write.write('Class 1: General way to write a text file\n')
f_write.write('After open, write file name and open mood\n')
f_write.write('Open mode w means write\n')

f_write.close()

#append file
f_append = open('readme.txt','a')
f_append.write('Open mode a means append\n')
f_append.write('Open mode r means append\n')

f_append.close()