def train_accuracy(Mat):
   Sum=0
   for i in Mat:
        if(i==1):
           Sum+=1.0
   return(Sum/len(Mat)*100)

def test_accuracy(Mat):
   Sum=0
   for i in Mat:
        if(i==-1):
           Sum+=1.0
   return(Sum/len(Mat)*100)
