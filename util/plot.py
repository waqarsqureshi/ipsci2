import matplotlib.pyplot as plt
import datetime

# Define the data for the tobacco and sesame datasets
tobacco_data = {
    1: {'images': 936, 'time': '3:30 pm', 'date': '07 April 2021'},
    2: {'images': 120, 'time': '3:38 pm', 'date': '07 April 2021'},
    3: {'images': 120, 'time': '5:52 pm', 'date': '07 April 2021'},
    4: {'images': 120, 'time': '6:33 pm', 'date': '07 April 2021'},
    5: {'images': 864, 'time': '2:30 pm', 'date': '09 April 2021'},
    6: {'images': 120, 'time': '3:27 pm', 'date': '09 April 2021'},
    7: {'images': 120, 'time': '3:43 pm', 'date': '09 April 2021'},
    8: {'images': 120, 'time': '3:59 pm', 'date': '09 April 2021'}
}

sesame_data = {
    1: {'images': 120, 'time': '6:30am', 'date': '09 August 2020'},
    2: {'images': 600, 'time': '8:30am', 'date': '09 August 2020'},
    3: {'images': 120, 'time': '11:30am', 'date': '10 August 2020'},
    4: {'images': 120, 'time': '6:00pm', 'date': '19 August 2020'},
    5: {'images': 120, 'time': '8:30am', 'date': '21 August 2020'},
    6: {'images': 600, 'time': '2:00pm', 'date': '21 August 2020'},
    7: {'images': 120, 'time': '3:30pm', 'date': '28 August 2020'},
    8: {'images': 120, 'time': '6:00pm', 'date': '06 September 2020'}
}

# Convert the date and time data into datetime objects
for key in tobacco_data:
    tobacco_data[key]['datetime'] = datetime.datetime.strptime(
        tobacco_data[key]['date'] + ' ' + tobacco_data[key]['time'], '%d %B %Y %I:%M %p'
    )

for key in sesame_data:
    sesame_data[key]['datetime'] = datetime.datetime.strptime(
        sesame_data[key]['date'] + ' ' + sesame_data[key]['time'], '%d %B %Y %I:%M%p'
    )

# Extract the datetime and images data for each dataset
tobacco_dates = [tobacco_data[key]['date'] for key in tobacco_data]
tobacco_time = [tobacco_data[key]['time'] for key in tobacco_data]
tobacco_images = [tobacco_data[key]['images'] for key in tobacco_data]

sesame_dates = [sesame_data[key]['date'] for key in sesame_data]
sesame_time = [sesame_data[key]['time'] for key in sesame_data]
sesame_images = [sesame_data[key]['images'] for key in sesame_data]

# Plot the scatter plot
print(tobacco_data[1]['time'])

plt.scatter(tobacco_dates, tobacco_time, tobacco_images, color='red', label='Tobacco')
plt.scatter(sesame_dates, sesame_time,sesame_images, color='blue', label='Sesame')
#plt.plot(tobacco_dates,tobacco_images)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Time')
plt.show()