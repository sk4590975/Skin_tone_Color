from django.http import HttpResponse
from django.shortcuts import render

def webpage1(request):
    return render(request,'index.html')

def webpage2(request):
    return render(request,'contact.html')

def webpage3(request):
    return render(request,'about.html')

def webpage4(request):
    return render(request,'form.html')
