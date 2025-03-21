import rosbag, sys, csv
import datetime

FILENAME = 'Indoor'
# SAVE_DIR = '/media/asimovsimpc/bulldog/aa-data/extracted_data/bulldog'
# BAGFILE = '/media/asimovsimpc/bulldog/temp_data/2023_11_24/compressd/2023_11_24_11_46_24.bag'
SAVE_DIR = '/home/pc1/Alpha Asimov/process_data/Data_saved/'
BAGFILE = '/home/pc1/Alpha Asimov/process_data/2023_11_21_16_38_03.bag'

def bag2csv(bag):
	#get list of topics from the bag
    listOfTopics = ["/ublox/fix", ]
    csv_filename = SAVE_DIR + 'GPS.csv'
    
    data_info = ["time", "lat", "long", "gps_err"]

    with open(csv_filename, 'w+') as csvfile:
        writer=csv.writer(csvfile, delimiter=',', lineterminator='\n',)
        writer.writerow(data_info)

        crr_data = {key: None for key in data_info}
        for topic in listOfTopics:
            extracted_data = bag.read_messages(topic)
            for seq_id, BagMessage in enumerate(extracted_data):  
                msg = BagMessage.message             
                secs = msg.header.stamp.secs 
                nsecs = msg.header.stamp.nsecs
                ms = '{:03d}'.format((int)(nsecs/1e6))
                lat = msg.latitude
                long = msg.longitude
                gps_error = (msg.position_covariance[0] + msg.position_covariance[4]) * 0.5
                date_time = datetime.datetime.fromtimestamp(secs).strftime("%Y_%m_%d_%H_%M_%S_") + ms
                

                # Append the data to the list
                sensor_data = [date_time, lat, long, gps_error]

                # Write the data to the CSV file
                writer.writerow(sensor_data)


if __name__ == '__main__':
    bag = rosbag.Bag(BAGFILE)
    bag2csv(bag)
    bag.close()

    print('PROCESS COMPLETE')