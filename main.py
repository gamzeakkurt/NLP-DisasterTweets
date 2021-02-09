#run train.py and test.py files

stream = open("train.py")
read_file = stream.read()
exec(read_file)


stream = open("test.py")
read_file = stream.read()
exec(read_file)
