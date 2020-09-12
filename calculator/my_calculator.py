# idle

import tkinter as tk
# tkinter : the standard python interface
# the module to use for GUI programming in python
# tkinter documentation
import time


num_list = ['7', '8', '9', '4', '5', '6', '1', '2', '3', '0', '.', '=']
op_list = ['*', '/', '+', '-', '(', ')', 'C', 'AC']

# creat a key list(num_lsit + op_list)
key_list = num_list + op_list

# remove =, C, AC in key_lsit
key_list.remove('=')
key_list.remove('AC')
key_list.remove('C')


# for using a keyboard
def inputKey(key):
    # display를 입력 가능한 상태로 전환
    display.configure(state=tk.NORMAL)
  
    print(key)
    # if inputKey in key_list
    if key.char in key_list:
        display.insert(tk.END, key.char)

    # if key is '=' or 'Enter'
    elif key.char == '=' or key.char == '\r':
        try:
            result = str(round(eval(display.get()), 2)) 
            display.delete(0, tk.END)      
            display.insert(tk.END, result)
        except:
            # save the incorrect text temporarily
            result_tmp = display.get()
            display.delete(0, tk.END)
            # message
            display.insert(0, 'cannot be calculated')
            display.update()

            # stop for a second
            time.sleep(1)

            display.delete(0, tk.END)

            # display result_tmp
            display.insert(0, result_tmp)

            
    # if key is 'C'
    elif key.char == 'C' or key.char == 'c':
        display.delete(0, tk.END)
    # if key is 'A'
    elif key.char == 'A' or key.char == 'a':
        display.delete(0, tk.END)
        clip1_entry.delete(0, tk.END)
        clip2_entry.delete(0, tk.END)
        clip3_entry.delete(0, tk.END)

    # if key is 'backspace'
    if key.keysym == 'BackSpace':
        # count current number of text
        display_len = len(display.get())
        # remove the last one
        display.delete(display_len-1, tk.END)

    # if key is 'F1'
    if key.keysym == 'F1':
        clip1_entry.delete(0, tk.END)
        clip1_entry.insert(tk.END, display.get())
    
    # if key is 'F2'
    elif key.keysym == 'F2':
        clip2_entry.delete(0, tk.END)
        clip2_entry.insert(tk.END, display.get())

    # if key is 'F3'
    elif key.keysym == 'F3':
        clip3_entry.delete(0, tk.END)
        clip3_entry.insert(tk.END, display.get())

    # if key is 'F4'
    elif key.keysym == 'F4':
        # clip-board 1's text append to display
        display.insert(tk.END, clip1_entry.get())
        # delete clip-board 1's text
        clip1_entry.delete(0, tk.END)

    # if key is 'F5'
    elif key.keysym == 'F5':
        display.insert(tk.END, clip2_entry.get())
        clip2_entry.delete(0, tk.END)
        
    # if key is 'F6'
    elif key.keysym == 'F6':
        display.insert(tk.END, clip3_entry.get())
        clip3_entry.delete(0, tk.END)

    # display를 입력 불가능한 상태로 전환
    display.configure(state='readonly')



# define "click"
def click(text):
    # display를 입력 가능한 상태로 전환
    display.configure(state=tk.NORMAL)
    # if text's value is '='
    if text == '=':
        try:
            # calculate a result
            result = str(round(eval(display.get()), 2)) # display.get() : display에 있는 글씨를 그대로 가져옴
                                                        # eval(입력값) : 입력값을 수식으로 보고 계산
                                                        # round(값, 2) : 값을 소수점 2자리로 반올
            display.delete(0, tk.END)      # display의 처음부터 끝까지 삭제
            display.insert(tk.END, result) # display의 가장 마지막에 result 값 추가
        except:
            # save the incorrect text temporarily
            result_tmp = display.get()
            display.delete(0, tk.END)
            # message
            display.insert(0, 'cannot be calculated')
            display.update()

            # stop for a second
            time.sleep(1)

            display.delete(0, tk.END)

            # display result_tmp
            display.insert(0, result_tmp)

   
    # if text's value is C
    elif text == 'C':
        # clear the main display
        display.delete(0, tk.END)
    elif text == 'AC':
        display.delete(0, tk.END)
    else:
    # if click the button, print a text of the button through Entry
        display.insert(tk.END, text)

    # display를 입력 불가능한 상태로 전환
    display.configure(state='readonly')

# if press F1-F6 button, run this function
def funcClick(key):
    # display를 입력 가능한 상태로 전환
    display.configure(state=tk.NORMAL)
    # F1 key
    if key == 'F1':
        # first, delete clip-board 1's text
        clip1_entry.delete(0, tk.END)
        # display Entry's text -> clip-board1
        clip1_entry.insert(tk.END, display.get())

    # F2 key
    elif key == 'F2':
        clip2_entry.delete(0, tk.END)
        clip2_entry.insert(tk.END, display.get())

    # F3 key
    elif key == 'F3':
        clip3_entry.delete(0, tk.END)
        clip3_entry.insert(tk.END, display.get())

    # F4 key
    elif key == 'F4':
        # clip-board 1's text append to display
        display.insert(tk.END, clip1_entry.get())
        # delete clip-board 1's text
        clip1_entry.delete(0, tk.END)

    # F5 key
    elif key == 'F5':
        display.insert(tk.END, clip2_entry.get())
        clip2_entry.delete(0, tk.END)
    # F6 key
    elif key == 'F6':
        display.insert(tk.END, clip3_entry.get())
        clip3_entry.delete(0, tk.END)

    # display를 입력 불가능한 상태로 전환
    display.configure(state='readonly')

                      
# create a window
window = tk.Tk()
window.mainloop()
window.title("MyCalculator")

# always on Top
window.attributes("-topmost", True)


# in focus-on, if press a key, run "inputKey"
window.bind("<Key>", inputKey)


# create a display
display = tk.Entry(window, width = 35, readonlybackground='light green', bg = 'light green') # entry : input window for a line
                                                           # parameters : where, width, color
display.grid(row=0, column=0, columnspan=2) # grid : where to place the Entry
display.configure(state='readonly')

# create a frame for number button
num_frame = tk.Frame(window)
num_frame.grid(row=1, column =0)

# create a number button

r = 0
c = 0
for btn_text in num_list:

    # define "cmd"
    def cmd(x=btn_text):
        click(x)

    tk.Button(num_frame, text = btn_text, width = 5, command = cmd).grid(row=r, column=c)
    c = c + 1
    if c > 2 :
        c = 0
        r = r+1


# create a frame for an operator button
op_frame = tk.Frame(window)
op_frame.grid(row=1, column =1)

# create an operator button

r = 0
c = 0

for btn_text in op_list :
    def cmd(x=btn_text):
        click(x)
    tk.Button(op_frame, text = btn_text, command = cmd, width = 5).grid(row=r, column=c)
    c = c+1
    if c > 1 :
        c = 0
        r = r+1

# create a clip-board frame
# clip-board?
# save a value in variable and load the value
clip_frame = tk.Frame(window)
clip_frame.grid(row=2, column=0, columnspan=2, sticky='N')

# F1-F6
def cmd_F1():
    # call funcClick simply
    funcClick('F1')

def cmd_F2():
    funcClick('F2')

def cmd_F3():
    funcClick('F3')

def cmd_F4():
    funcClick('F4')

def cmd_F5():
    funcClick('F5')

def cmd_F6():
    funcClick('F6')

    


# clip board 1: input-output button
clip1_input_btn = tk.Button(clip_frame, width=2, text='F1', command=cmd_F1)
clip1_input_btn.grid(row=0, column=0)
clip1_entry = tk.Entry(clip_frame, width=20, bg='light pink')
clip1_entry.grid(row=0, column=1)
clip1_output_btn = tk.Button(clip_frame, width=2, text='F4', command=cmd_F4)
clip1_output_btn.grid(row=0, column=2)



clip2_input_btn = tk.Button(clip_frame, width=2, text='F2', command=cmd_F2)
clip2_input_btn.grid(row=1, column=0)
clip2_entry = tk.Entry(clip_frame, width=20, bg='light blue')
clip2_entry.grid(row=1, column=1)
clip2_output_btn = tk.Button(clip_frame, width=2, text='F5', command=cmd_F5)
clip2_output_btn.grid(row=1, column=2)


clip3_input_btn = tk.Button(clip_frame, width=2, text='F3', command=cmd_F3)
clip3_input_btn.grid(row=2, column=0)
clip3_entry = tk.Entry(clip_frame, width=20, bg='light yellow')
clip3_entry.grid(row=2, column=1)
clip3_output_btn = tk.Button(clip_frame, width=2, text='F6', command=cmd_F6)
clip3_output_btn.grid(row=2, column=2)

        

# connect button to window : Event Handling
# Event : a signal generated by the user's action
# e.g. starting a menu, using a keyboard, pressing a button etc.

# user's action -> generating an event -> processing an event -> feedback -> user's action
# for connection, create "click"







