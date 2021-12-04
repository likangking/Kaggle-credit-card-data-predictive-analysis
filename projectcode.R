library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(VIM)
library(randomForest)
library(class)
library(pROC)

#读数据
card <- read.csv('../Data/creditcard.csv')       
card <- as.data.frame(card)                  
str(card)                                     
summary(card)                            


prop.table(table(card$Class))      

aggr(card, prop = F, numbers = T)   
which(is.na(card$Time) == T)        
card[153756:153762, 1:3]            
card[153759, 1] <- 100001            

card$Time_Hour <- round(card[, 1]/3600, 0)
card$Class <- factor(card$Class)           

card_1 <- card[card$Class == '1', ]       
card_0 <- card[card$Class == '0', ]        


set.seed(1234) 
index <- sample(x = 1:nrow(card_0), size = nrow(card_1))
card_0_new <- card_0[index, ] 
card_end <- rbind(card_0_new, card_1) 


card_end <- card_end[-1] %>% select(Time_Hour, everything())

## 分层抽样，建立训练集和测试集
set.seed(1234) 

## 按照新数据的目标变量进行8:2的分层抽样，返回矩阵形式的抽样索引
index2 <- createDataPartition(card_end$Class, p = 0.8, list = F)
traindata <- card_end[index2, ] # 创建训练集
testdata <- card_end[-index2, ] # 创建测试集

## 验证
table(card_end$Class)
table(traindata$Class)
table(testdata$Class)


## 对数据进行标准化
standard <- preProcess(card_end, method = 'range')
card_s <- predict(standard, card_end)
traindata2 <- card_s[index2, ]
testdata2 <- card_s[-index2, ]




# 不同时间诈骗次数条形图
ggplot(card_1, aes(x = factor(Time_Hour), fill = factor(Time_Hour))) + 
  geom_bar(stat = 'count') +            
  theme_minimal() +                    
  labs(x = 'Time_Hour', y = 'Count') + 
  theme(legend.position = 'none')      


# 不同时间诈骗金额箱线图
ggplot(card_1, aes(x = factor(Time_Hour), y = Amount, fill = factor(Time_Hour))) + 
  geom_boxplot() +
  theme_minimal() +
  labs(x = 'Time_Hour', y = 'Mean') +
  theme(legend.position = 'none')


# 获取下面条形图所需的数据源
df_card_1 <- 
  card_1 %>%
  group_by(Time_Hour) %>%             
  summarise(Mamount = mean(Amount))   

# 不同时间平均诈骗金额条形图
ggplot(df_card_1, aes(x = factor(Time_Hour), y = Mamount, fill = factor(Time_Hour))) + 
  geom_bar(stat = 'identity') +      
  theme_minimal() +             
  labs(x = 'Time_Hour', y = 'Mean_Amount') +
  theme(legend.position = 'none')


## 5折交叉验证的方法建立随机森林模型
set.seed(1234)
model_rf <- train(Class ~., data = traindata, method = 'rf', 
                  trControl = trainControl(method = 'cv', 
                                           number = 5, 
                                           selectionFunction = 'oneSE'))
model_rf
pred_rf <- predict(model_rf, testdata[-31])              # 预测

# 建立混淆矩阵
confusionMatrix(data = pred_rf, reference = testdata$Class, positive = '1')
plot(varImp(model_rf))                              


## knn
results = c()    
for(i in 3:10) {
  set.seed(1234)
  pred_knn <- knn(traindata2[-31], testdata2[-31], traindata2$Class, i)
  Table <- table(pred_knn, testdata$Class)
  accuracy <- sum(diag(Table))/sum(Table) 
  results <- c(results, accuracy)
}


plot(x = 3:10, y = results, type = 'b', col = 'blue', xlab = 'k', ylab = 'accuracy')

set.seed(1234)


pred_knn <- knn(train = traindata2[-31], test = testdata2[-31], 
                cl = traindata2$Class, k = 4)

confusionMatrix(pred_knn, testdata2$Class, positive = '1')

## 数据框（knn预测值、随机森林预测值、class）
pred_results <- data.frame(knn = pred_knn, rf = pred_rf, class = testdata$Class)
index3 <- which(pred_results$knn != pred_rf)
pred_results[index3, ]                      
write.csv(pred_results[index3, ],file = "../Doces/modeldata.csv")

