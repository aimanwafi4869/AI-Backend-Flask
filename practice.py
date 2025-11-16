# import tensorflow as tf

# # Avoid maximum Memory Allocation
# # devices = tf.config.list_physical_devices('GPU') 
# # tf.config.experimental.set_memory_growth(devices[0], True)

# #Initialize
# x = tf.constant(4,shape=(1,1), dtype=tf.float32)
# x = tf.constant([[1,2,3],[4,5,6]])

# x = tf.ones((3,3))
# x = tf.zeros((2,3))
# x = tf.eye(3) #Identity Matrix
# x = tf.random.normal((3,3), mean=0, stddev=1) # Random Distribution
# x = tf.random.uniform((1,3), minval=0, maxval=1) # Normal Distribution
# x = tf.range(start=1, limit=10, delta=2) #Delta is increment
# x = tf.cast(x, dtype=tf.float64)

# #Math Operation

# #Initialize
# x = tf.constant([[1,2,3]])
# y = tf.constant([[9,8,7]])

# z = tf.add(x,y) #Addition
# z = tf.subtract(x,y) #Substraction
# z = tf.multiply(x,y) #Multiplication
# z = tf.divide(x,y) #Division
# z = x + y # Alt. Addition
# z = x - y # Alt. Substraction
# z = x * y # Alt. Multiplication
# z = x / y # Alt. Division

# z = x ** 2 #Exponentiate each element by 2 (Power each elemtent by 2)

# x = tf.constant([[1,2,3]])
# y = tf.constant([[9],[8],[7]])
# z = tf.tensordot(x, y, axes=1)
# z = tf.reduce_sum(x*y, axis=0)

# x = tf.random.normal((2,3))
# y = tf.random.normal((3,4))
# z = tf.matmul(x,y) #matrix multiply
# z = x @ y #matrix multiply

# #Indexing
# x = tf.constant([0,1,2,3,4,5,6])
# # print(x[:]) #[0,1,2,3,4,5,6] (print all)
# # print(x[1:]) #[1,2,3,4,5,6] (start from index 1)
# # print(x[1:3]) #[1,2] (start from index 1 end before index 3)
# # print(x[::2]) #[0,2,4,6] (increment by 2)
# # print(x[::-1]) #[6,5,4,3,2,1,0] (Reverse)

# index = tf.constant([0,3,5])
# x_ind = tf.gather(x,index) # Search the same number as index (above)

# x = tf.constant([[1,2],
#                  [3,4],
#                  [5,6]])
# # print(x[0,:]) #[1,2] seperate dimension by using , (comma) 
# # print(x[0:2,]) #[1,2],[3,4] start from index 0 end before index 2

# #Reshaping

# x = tf.range(9)
# print(x) # [0 1 2 3 4 5 6 7 8]

# x = tf.reshape(x, (3,3))
# print(x)

# x = tf.transpose(x,perm=[1,0])
# print(x)

# print(len({1: 'a', 2: 'b', 3: 'c'}))
# _asd = "avx"
# print(_asd)
# print("Python".startswith('P'))
# print(f"Result: {10+5}")

# try:
#     if x == 10:
#         print("x is equal to 10")
# except Exception as e:
#     print(e)


# x = 14
# if x > 5:
#     print("x is greater than 5")
# # Executed if condition one is True.
# if x < 10:
#     print("x is less than 10")
# # Executed if condition two is True.
# if x == 10:
#     print("x is equal to 10")
# # Executed if condition three is True.
# else:
#     print('else will be executed')

# ready = "Y"
# print(ready == ("Yes" or "Y" or "y"))
# name = input("What is your name? ")
# birthYear = input("What year were you born in? ")


# birthYear = birthYear.replace('k','0')

# birthYear = int(birthYear)
# if birthYear >= 2012:
#     print(name,", you are a member of Generation Alpha", sep='213', end='9999')

# a = input("A: ")
# # b = input("B: ")
# # c = input("C: ")

# # if a == b:
# #     if b == c:
# #         print('Wow what are the odds')
# #     else:
# #         print('They wont all be the same')
# # # Executed if condition three is True.
# # else:
# #     print('They wont all be the same')
# match a:
#     case 'a':
#         print('A')
#     case 'b':
#         print('B')

# a = 'babbabababa'
# for i in range(10,2,-2):
#     print(a)
# else:
#     print('vavasv')

# fruits = ['apples','banana','cherry']

# for x in fruits:
#     print(x)
#     if x == 'banana':
#         break

# for x in fruits:
#     if x == 'banana':
#         break
#     print(x)

# numbers = []

# num_count = int(input('How many numbers ? '))
# for i in range(num_count):
#     num = int(input('Enter number : '))
#     numbers.append(num)

# smallest = numbers[0]

# for num in numbers:
#     if num < smallest:
#         smallest = num

# print('The smallest number is ', smallest)


#Exercise 1 - Sum all number 1 - n
# n = int(input('Enter number : '))
# sum = 0
# for i in range(1,n+1):
#     sum += i
# print(sum)



#Exercise 2 - Count Vowels in a sentence
# word = input('Enter any word : ').lower()
# totalVowel = 0
# vowelA = 0
# vowelE = 0
# vowelI = 0
# vowelO = 0
# vowelU = 0

# for w in word:
#     if w == 'a':
#         vowelA+=1
#         totalVowel+=1
#     elif w == 'e':
#         vowelE+=1
#         totalVowel+=1
#     elif w == 'i':
#         vowelI+=1
#         totalVowel+=1
#     elif w == 'o':
#         vowelO+=1
#         totalVowel+=1
#     elif w == 'u':
#         vowelU+=1
#         totalVowel+=1

# print('Vowel A :', vowelA)
# print('Vowel E :', vowelE)
# print('Vowel I :', vowelI)
# print('Vowel O :', vowelO)
# print('Vowel U :', vowelU)
# print('Total Vowel', totalVowel)

# #Exercise 3 - Multiplication Table for n * 1 - 10
 
# n = int(input('Enter number : '))
# for i in range(1,11):
#     print(f'{n} x {i} = {n * i}')


#Exercise 4 - Reverse String
 
# print(input('Enter any word : ')[::-1])
# text = input('Enter any word : ')
# reverse = ""
# for i in range(1, len(text) + 1):
#     reverse += text[len(text) - i]

# print(reverse)


# n = 1

# while n <= 5:
#     print(str(n) * n)
#     n+=1



# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import pandas as pd

# # Load the Iris dataset
# # data = load_iris()
# # X = data.data  # Features
# # y = data.target  # Labels

# dataset = pd.read_csv('/home/aiman/Documents/FlaskApp/dataset/dataset(Hackathon).csv')
# # print(dataset.head())
# xData = dataset.iloc[:,1:]
# yData = dataset.iloc[:,0:1]
# # print(xData)
# # print(yData)

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=100)

# # Initialize the Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=100)

# # Train the model
# rf_classifier.fit(X_train, y_train)

# # Make predictions
# y_pred = rf_classifier.predict(X_test)
# # print(X_test)
# # print(y_test)
# # print(y_pred)
# print(X_test)
# print('\n\n\n\n\n\n')
# print(y_pred)
# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Exercise 1 - Countdown Timer



# Exercise 2 - Password Checker
# password = 'python123'
# saman = 0 
# while True:
#     inputPassword = input('Enter password : ')

#     if inputPassword == password:
#         break
#     saman+=10
#     print('Saman 10 ringgit')

# print('Correct Password - Jumlah Saman RM',saman)


i = 0
while i < 4:
    j = 0
    while j < 4:
        print('*',end='')
        j+=1
    print()
    i+=1

max = 5
for a in range(max):
    for b in range(a):
        print('*',end='')
    print()
