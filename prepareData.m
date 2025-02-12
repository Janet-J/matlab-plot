clear; clc
[y, Fs] = audioread("audio/original.wav");
save original.mat
W = y;
clear y
W(100000:122050) = NaN;
save stegoAudio.mat
display("Done")
clear; clc
