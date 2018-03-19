from os import listdir
from os.path import isfile, join

path = "/Users/Pedro/Desktop/csv/TT100K/valid_txt"
imagesPath = "/Users/Pedro/Desktop/csv/TT100K/valid"

def main():
	files = listdir(path)
	output = open("/Users/Pedro/Desktop/csv/TT100K/annotations_valid.csv","w")
	count = []
	max = 0
	for i in range(0,45):
		count.append(0)
	for file in files:
		print file
		F = open( path + "/" + file,"r")
		for line in F:
			line = line.split() 
			i = int(line[0])
			count[i] += 1
			#output.write(imagesPath + "/" + imagefile + "," + line[1] + "," + line[2] + "," + line[3] + "," + line[4] +"," + line[0] +"\n")
		F.close() 
	output.close()

	print count





if __name__ == "__main__":
	main()