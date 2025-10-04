from pose import *
import os
import sys
import cv2
import json
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


if 'output.json' in os.listdir():
    # Read output.json
    last_process = json.load(open('output.json', 'r'))
    out_dict = {'pose_output': []}
    for l_pose in last_process['pose_output']:
        l_address = l_pose['address']
        l_name = l_pose['name']
        l_format = l_pose['format']

        # Check files existent
        if os.path.isfile(f'{l_address}{l_name}.{l_format}') and os.path.isfile(f'{l_address}{l_name} out.{l_format}')\
                and l_pose not in out_dict['pose_output']:
            out_dict['pose_output'].append(l_pose.copy())

else:  # output.json is not found
    out_dict = {'pose_output': list()}

gest = Pose()


def process_image(file_name, format, address=""):

    # Read the image
    img = cv2.imread(f'{address}{file_name}.{format}')

    # Process & save the output
    global gest
    cv2.imwrite(f'{address}{file_name} out.{format}', gest(img))
    cv2.imshow('output', cv2.imread(f'{address}{file_name} out.{format}'))
    tk.messagebox.showinfo('خروجی ذخیره شد', '.برنامه با موفقیت اجرا شد\nمحل ذخیره فایل ورودی را برای مشاهده خروجی، '
                                             '.بررسی کنید')


def process_video(file_name, format, address=""):

    # Preprocessing
    cap = cv2.VideoCapture(f'{address}{file_name}.{format}')  # Read video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out_vid = cv2.VideoWriter(f'{address}{file_name} out.{format}', cv2.VideoWriter_fourcc(*'MP4V'), fps,
                              (frame_width,frame_height))  # Output video

    global gest
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            # Process with pose object
            out = gest(frame.copy())

            # Show and Write the output
            cv2.imshow(file_name+" out", out)
            out_vid.write(out)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # Close all
    out_vid.release()
    cap.release()
    cv2.destroyAllWindows()
    tk.messagebox.showinfo('خروجی ذخیره شد', '.برنامه با موفقیت اجرا شد\nمحل ذخیره فایل ورودی را برای مشاهده خروجی، '
                                             '.بررسی کنید')


def previous_tries():
    # Check whether we have any previous tries or not
    if len(out_dict['pose_output']) == 0:
        tk.messagebox.showwarning('ناموجود', '.در حافظه وجود ندارد Correct GeSit نتیجه هیچ پردازشی در بخش \n'
                                             '.لطفا ابتدا یک پردازش در آن بخش انجام دهید')
        return

    # root of previous tries page
    root4 = tk.Tk()
    root4.option_add('*Dialog.msg.font', 'B Elm')
    root4.title('نتایج پردازش های قبلی')

    # Make previous tries page
    yscrollbar = tk.Scrollbar(root4)
    yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tk.Label(root4, text="لطفا نام فایل مورد نظر را برای\n.مشاهده نتیجه آن، انتخاب کنید", font=('Calibri', 14)).pack()

    # Make our list
    list = tk.Listbox(root4, selectmode="multiple", yscrollcommand=yscrollbar.set)
    list.pack(padx=10, pady=10, expand=tk.YES, fill="both")

    # Check items
    tries = []
    deleted = []
    for t in out_dict['pose_output']:  # Check files existent
        if os.path.isfile(f"{t['address']}{t['name']}.{t['format']}") and os.path.isfile(f"{t['address']}{t['name']} out."
                                                                                         f"{t['format']}"):
            tries.append(t)
        else:
            deleted.append(t)

    # Remove deleted file from out_dict
    if len(deleted):
        for delete in deleted:
            out_dict['pose_output'].remove(delete)
    del deleted

    # Again check whether we have any previous tries or not
    if len(out_dict['pose_output']) == 0:
        tk.messagebox.showwarning('ناموجود', '.در حافظه وجود ندارد Correct GeSit نتیجه هیچ پردازشی در بخش \n'
                                             '.لطفا ابتدا یک پردازش در آن بخش انجام دهید')
        root4.destroy()
        return

    # Add items to the list
    for each_item in range(len(tries)):
        list.insert(tk.END, tries[each_item]['name'])
        list.itemconfig(each_item, bg="lime")

    # Function to show tries
    def check_tries():
        for i in list.curselection():
            t = tries[i]
            if t['format'] in ('png', 'jpg', 'jpeg'):  # Show image tries
                cv2.imshow(f"{t['name']} input", cv2.imread(f"{t['address']}{t['name']}.{t['format']}"))
                cv2.imshow(f"{t['name']} output", cv2.imread(f"{t['address']}{t['name']} out.{t['format']}"))
            elif t['format'] == 'mp4':  # Show video tries
                cap = cv2.VideoCapture(f'{t["address"]}{t["name"]} out.{t["format"]}')
                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret is True:
                        cv2.imshow(f"{t['name']} output", frame)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    # Break the loop
                    else:
                        break
                cap.release()

    yscrollbar.config(command=list.yview)
    ttk.Button(root4, text="نمایش نتایج", command=check_tries).pack()
    tk.Button(root4, text="بستن", bg='red', command=root4.destroy, font=('B Elm', 9)).pack()
    root4.mainloop()


def pose_help():
    # root and frame of pose help page
    root6 = tk.Tk()
    root6.title('Correct GeSit راهنمای')
    poseH_frm = ttk.Frame(root6, padding=10)
    poseH_frm.grid()

    # Make the pose help page
    tk.Label(poseH_frm, text='Correct GeSit راهنمای بخش', font=('Calibri', 22)).grid(column=0, row=0)
    tk.Label(poseH_frm).grid(column=0, row=1)

    tk.Label(poseH_frm, text='برای اجرای صحیح برنامه، آدرس فایل ورودی را', font=('Calibri', 18)).grid(column=0, row=2)
    tk.Label(poseH_frm, text='.دقیق و درست وارد کنید', font=('Calibri', 18)).grid(column=0, row=3)
    tk.Label(poseH_frm, text='،mp4 ،jpg دقت کنید که فقط فایل های', font=('Calibri', 18)).grid(column=0, row=4)
    tk.Label(poseH_frm, text='.قابل پردازش می باشند png و jpeg', font=('Calibri', 18)).grid(column=0, row=5)
    tk.Label(poseH_frm).grid(column=0, row=6)

    tk.Label(poseH_frm, text='.قطعا دقت مدل 100 درصد نیست و اشتباهاتی دارد', font=('Calibri', 18)).grid(column=0, row=7)
    tk.Label(poseH_frm, text='در صورت نگرفتن نتیجه مطلوب، زاویه دوربین را ', font=('Calibri', 18)).grid(column=0, row=8)
    tk.Label(poseH_frm, text='.کمی تغییر دهید', font=('Calibri', 18)).grid(column=0, row=9)
    tk.Label(poseH_frm, text='', font=('Calibri', 18)).grid(column=0, row=10)

    tk.Label(poseH_frm, text='همچنین توجه داشته باشید که صورت شخص، حتما', font=('Calibri', 18)).grid(column=0, row=11)
    tk.Label(poseH_frm, text='.باید در تصویر واضح باشد', font=('Calibri', 18)).grid(column=0, row=12)
    tk.Label(poseH_frm).grid(column=0, row=13)

    tk.Label(poseH_frm, text='ممکن است هنگام پردازش فیلم، به سیستم فشار', font=('Calibri', 18)).grid(column=0, row=14)
    tk.Label(poseH_frm, text='.بیاید و عملکرد صحیح نداشته باشد', font=('Calibri', 18)).grid(column=0, row=15)
    tk.Label(poseH_frm).grid(column=0, row=16)
    tk.Button(poseH_frm, text='بستن', bg='red', command=root6.destroy, font=('B Elm', 12)).grid(column=0, row=17)
    root6.mainloop()


def call_us():
    # root and frame for about us page
    root3 = tk.Tk()
    root3.title('درباره ما')
    us_frm = ttk.Frame(root3, padding=10)
    us_frm.grid()

    # Make about us page
    tk.Label(us_frm, text='برنامه نویس: مهسان محمدزاده', font=('Calibri', 18)).grid(column=0, row=0)
    tk.Label(us_frm, text='mahsan.ms1386@gmail.com :تماس با برنامه نویس از طریق', font=('Calibri', 18)).grid(column=0,
                                                                                                            row=1)

    tk.Button(us_frm, text='بستن', command=root3.destroy, bg='red', font=('B Elm', 12)).grid(column=0, row=2)
    root3.mainloop()


def Exit():
    json.dump(out_dict, open('output.json', 'w'))  # Save out_dict to output.json
    sys.exit()


def pose():
    # root and frame of pose page
    root1 = tk.Tk()
    root1.option_add('*Dialog.msg.font', 'B Elm')
    root1.title('Correct GeSit')
    pose_frm = ttk.Frame(root1, padding=10)
    pose_frm.grid()

    # Make pose page
    tk.Label(pose_frm, text='<تشخیص کیفیت نشستن>', font=('B Mahsa', 20, 'bold')).grid(column=0, row=0)
    tk.Label(pose_frm, text=".آدرس و نام فایل ورودی را وارد کنید", font=('Calibri', 15)).grid(column=0, row=1)
    ans = tk.Entry(pose_frm)  # Get a text input
    ans.grid(column=0, row=2)

    # Function to run the right function for image or video
    def do(file_name, address, format):
        data_dict = {'address': address,
                     'name': file_name,
                     'format': format}

        # Check file existent
        if not os.path.isfile(f'{address}{file_name}.{format}'):
            tk.messagebox.showerror('وجود فایل', '.فایل مورد نظر، وجود ندارد')
            return

        # The file is an image
        if format in ('png', 'jpg', 'jpeg'):
            try:
                process_image(file_name, format, address=address)
            except:
                tk.messagebox.showerror('ارور', '.اروری هنگام اجرای برنامه پیش آمد\nاگر صورت شخص به طور کامل در تصویر '
                                                'معلوم نیست، لطفا تصویر .دیگری را برای اجرا انتخاب کنید\n.یا ممکن است '
                                                'مشکل از کیفیت پایین تصویر باشد')
            else:  # Add new file to out_dict
                if data_dict not in out_dict['pose_output']:
                    out_dict['pose_output'].append(data_dict)

        # The file a video
        elif format == 'mp4':
            try:
                process_video(file_name, format, address=address)
            except:
                tk.messagebox.showerror('ارور', '.اروری هنگام اجرای برنامه پیش آمد\nاگر صورت شخص به طور کامل در ویدئو '
                                                'معلوم نیست، لطفا ویدئو دیگری را برای اجرا انتخاب کنید\n.یا ممکن است '
                                                'مشکل از کیفیت پایین ویدئو باشد')
            else:  # Add new file to out_dict
                if data_dict not in out_dict['pose_output']:
                    out_dict['pose_output'].append(data_dict)

        # The format is not what we expected, so better not to process it
        else:
            tk.messagebox.showwarning('فرمت ورودی', 'jpg, jpeg, png, mp4 تنها فرمت های قابل پردازش برای برنامه '
                                                    '\n.هستند\n.لطفا فرمت فایل ورودی را تغییر دهید')

    def check_name():
        file_name, address, format = (None, None, None)
        data = ans.get()
        # Separate file name if it was address
        try:
            input_ad, format = data.split('.')  # Raise ValueError if the string has no dot or more than 2 dots.
            input_ad = input_ad.split("\\")
            if len(input_ad) == 1:  # The file is in current directory.
                file_name = str(input_ad[0])
                address = ""
            else:  # The file is in another directory
                file_name = input_ad[-1]
                address = input_ad[0]
                for ad in input_ad[1:-1]:
                    address += f"\\{ad}"
                address += "\\"
            # Splitted file name, address and format successfully, so run the correct function for image or video
            do(file_name, address, format)

        except ValueError:  # User haven't entered format!
            tk.messagebox.showerror('ارور', '.لطفا نام فایل را کامل، درست و به همراه پسوند آن وارد کنید')

    ttk.Button(pose_frm, text='ثبت', command=check_name).grid(column=0, row=3)
    tk.Button(pose_frm, text='مشاهده نتایج پردازش های قبلی', command=previous_tries, bg='green', fg='white',
              font=('B Elm', 10)).grid(column=0, row=4)
    tk.Button(pose_frm, text='راهنما', command=pose_help, bg='cyan', font=('B Elm', 7)).grid(column=0, row=5)
    tk.Button(pose_frm, text='تماس با ما', command=call_us, bg='magenta', font=('B Elm', 8)).\
        grid(column=0, row=6)
    tk.Button(pose_frm, text='خروج', bg='red', command=Exit, font=('B Elm', 8)).grid(column=0, row=7)
    root1.mainloop()


if __name__ == '__main__':
    pose()
