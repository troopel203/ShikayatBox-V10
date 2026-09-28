# Trains TF-IDF-free char-ngram Naive Bayes on SYNTHETIC complaints (EN/HI/MR/Hinglish).
# Replace seeds with real complaint data when available. Exports model.json for the browser.
import json,random,itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
random.seed(7)
S={
'road':["big pothole on the road near {p}","road is broken with deep potholes at {p}","road damaged cracks dangerous for bikes {p}","{p} pe sadak me gadda hai","sadak toot gayi hai {p} ke paas","speed breaker broken road {p}","रस्त्यावर मोठा खड्डा आहे {p}","सड़क पर बड़ा गड्ढा है {p} के पास","रस्ता खराब झाला आहे {p}","footpath tiles broken {p}"],
'garbage':["garbage dump not collected for days near {p}","plastic waste piled up at {p}","trash overflowing dustbin {p}","{p} pe kachra pada hai bahut smell","kachra gadi nahi aayi {p}","रस्त्यावर कचरा साचला आहे {p}","कचरा उचलला नाही {p} जवळ","यहाँ कूड़ा और प्लास्टिक पड़ा है {p}","waste burning litter everywhere {p}","कचरा कुंडी भरून वाहते आहे {p}"],
'drainage':["drain blocked and sewage overflowing at {p}","manhole open without cover near {p}","gutter choked waterlogging {p}","{p} me nali jam hai gandha pani","naali band hai pani bhar gaya {p}","गटार तुंबले आहे {p} येथे","नाला जाम है गंदा पानी सड़क पर {p}","ड्रेनेज चोकअप झाले आहे {p}","sewer line overflow smell {p}","open manhole dangerous {p}"],
'light':["street light not working near {p}","streetlight broken dark at night {p}","lamp post pole wire hanging {p}","{p} pe street light band hai andhera","light kharab hai raat ko andhera {p}","पथदिवा बंद आहे {p} येथे","स्ट्रीट लाइट बंद है अंधेरा {p}","खांबावरचा दिवा बंद आहे {p}","electric pole leaning wires exposed {p}","विजेचा खांब वाकला आहे {p}"],
'water':["water pipeline leaking at {p}","pipe burst water wasting on road {p}","no water supply since morning {p}","{p} pe paani ki pipe leak ho rahi hai","nal me paani nahi aa raha {p}","पाण्याची पाईप फुटली आहे {p}","पाईपलाईन लीक होत आहे {p} जवळ","पानी की पाइप फट गई है {p}","tap water leakage wastage {p}","पाणीपुरवठा बंद आहे {p}"],
'animal':["dead dog lying on road near {p}","injured cow needs help at {p}","stray animal hurt bleeding {p}","{p} pe kutta mara pada hai","ghayal gai sadak pe {p}","मेलेला कुत्रा रस्त्यावर पडला आहे {p}","जखमी गाय {p} जवळ मदत हवी","घायल कुत्ता सड़क पर पड़ा है {p}","dead cat animal carcass smell {p}","bull injured on highway {p}"]}
P=["Kothrud","Hadapsar","Shivajinagar","Baner","Katraj","Viman Nagar","Camp","Wakad","school","hospital","bus stop","market","temple","चौक","स्टेशन"]
X,y=[],[]
for c,ts in S.items():
  for t in ts:
    for p in P:
      s=t.format(p=p);X.append(s);y.append(c)
      X.append(s.lower().replace("the ","")+" please fix");y.append(c)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=1,stratify=y)
v=CountVectorizer(analyzer='char_wb',ngram_range=(2,4),max_features=4000,lowercase=True)
m=MultinomialNB(alpha=.3).fit(v.fit_transform(Xtr),ytr)
print("held-out accuracy:",round(m.score(v.transform(Xte),yte),3),"(synthetic, templates overlap -> optimistic)")
voc=v.get_feature_names_out().tolist()
json.dump({"classes":m.classes_.tolist(),"prior":[round(x,3) for x in m.class_log_prior_.tolist()],
 "vocab":voc,"logp":[[round(x,2) for x in r] for r in m.feature_log_prob_.tolist()]},open("model.json","w"),ensure_ascii=False,separators=(',',':'))
