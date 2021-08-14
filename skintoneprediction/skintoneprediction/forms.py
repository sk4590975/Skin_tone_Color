from django import forms

from .models import Boneage

class BoneageForm(forms.ModelForm):
    class Meta:
        model = Boneage
        fields = ('gender', 'age','images')
