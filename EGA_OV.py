import xlrd
import xlwt
import os
import pandas as pd
import csv
# pip install scikit-opt
import sko.GA as GA
import random

# 车身对象，需要明确其每一秒所在的具体位置
class car(object):
    def __init__(self, Id, modelType, powerType, driverType):
        # 环境信息
        self.Id = Id  # 唯一确定ID，方便输出结果
        self.location = 'inPort'  # 车身目前在何处
        self.pathId = None  # 在哪个车道，返回道为-1
        self.parkId = None  # 若在车道上，在几号停车位
        self.moveTime = None  # 若在车道上，已移动了多久
        # 车身自带信息
        self.modelType = modelType
        self.powerType = powerType
        self.driverType = driverType

    def getLocation(self):
        if self.location == 'ePath':
            return str(self.pathId + 1) + str(self.parkId + 1)
        elif self.location == 'bPath':
            return str(7) + str(self.parkId + 1)
        elif self.location == 'inPort':
            return str(0)
        elif self.location == 'lMover':
            return str(1)
        elif self.location == 'rMover':
            return str(2)
        elif self.location == 'outPort':
            return str(3)
        else:
            return ''


class inPort(object):
    def __init__(self, carPool):
        self.carList = carPool

    # 由Mover调用，从入口取走一个车身
    def getFrontCar(self):
        if len(self.carList):
            self.carList[0].location = 'lMover'
            return self.carList.pop(0)
        else:
            return None


class outPort(object):
    def __init__(self, carNum):
        self.sequence = []
        self.carNum = carNum

    # 由Mover调用，将车身送入出口
    def enter(self, car):
        self.sequence.append(car)
        car.location = 'outPort'

    # 由Env调用，查询是否完成所有调度
    def finished(self):
        return len(self.sequence) == self.carNum


class Paths(object):
    def __init__(self):
        # 进车道的车身情况
        self.ePath = [[None] * 10 for _ in range(6)]
        # 返回道的车身情况
        self.bPath = [None] * 10

    # 由Mover调用，将车身装载在车道上
    def loadCar(self, car, pathId):
        # check
        if pathId == -1 and self.bPath[0] is not None:
            return False
        if pathId != -1 and self.ePath[pathId][9] is not None:
            return False
        #
        if pathId == -1:
            self.bPath[0] = car
            car.location = 'bPath'
            car.pathId = -1
            car.parkId = 0
            car.moveTime = 0
        else:
            self.ePath[pathId][9] = car
            car.location = 'ePath'
            car.pathId = pathId
            car.parkId = 9
            car.moveTime = 0
        return True

    # 由Mover调用，取下车身并返回给Mover
    def unloadCar(self, pathId):
        car = None
        if pathId == -1:
            car = self.bPath[9]
            if car is None:
                return car
            car.location = 'lMover'
            self.bPath[9] = None
        else:
            car = self.ePath[pathId][0]
            if car is None:
                return car
            car.location = 'rMover'
            self.ePath[pathId][0] = None
        return car

    # 内置函数
    def canMove(self, car):
        pathid = car.pathId
        parkid = car.parkId
        if pathid == -1 and parkid != 9:
            return self.bPath[parkid + 1] is None
        elif pathid != -1 and parkid != 0:
            return self.ePath[pathid][parkid - 1] is None
        return False

    # 内置函数
    def moveCar(self, car):
        car.moveTime += 1
        if car.moveTime == 9:
            car.moveTime = 0
            if self.canMove(car):
                pathid = car.pathId
                parkid = car.parkId
                if pathid == -1:
                    car.parkId += 1
                    self.bPath[parkid] = None
                    self.bPath[car.parkId] = car
                else:
                    car.parkId -= 1
                    self.ePath[pathid][parkid] = None
                    self.ePath[pathid][car.parkId] = car

    # 每秒种调用一次
    def moveCars(self):
        for path in self.ePath:
            for car in path:
                if car is not None:
                    self.moveCar(car)
        for car in self.bPath:
            if car is not None:
                self.moveCar(car)

    # 由Mover调用
    def getFront(self, pathId):
        if pathId == -1:
            return self.bPath[9]
        else:
            return self.ePath[pathId][0]


class lMover(object):
    def __init__(self, inPort, Paths):
        self.inPort = inPort  # 指向连接的输入口
        self.Paths = Paths  # 指向连接的车道组
        self.idle = True  # 是否空闲
        self.carry = None  # 是否装着那辆车
        self.target = None  # 是否在往哪个车道走
        self.leftTime = 0  # 如果在移动，还需要多久
        self.loadCost = [9, 6, 3, 0, 6, 9]  # 将车身送入进车道的耗时
        self.transCost = [12, 9, 6, 3, 6, 9]  # 将车身从返回道再度送入进车道的耗时
        self.wait = 0

    def execute(self, source, target):
        # Env每一秒都向Mover调用该函数
        # 返回True代表这条指令被完成
        # 装车需要考虑是否堵着，接车需要考虑是否到了
        # 指令格式：['inPort'/'bPath'] [pathid],
        if source == 'wait':
            if self.idle:
                self.idle = False
                self.wait = self.loadCost[target] - 1
                return False
            else:
                self.wait -= 1
                if self.wait != 0:
                    return False
                else:
                    self.idle = True
                    return True

        if self.idle:
            self.pickup(source, target)
        if not self.idle and self.leftTime == 0:
            if self.arrive():
                return True
        if self.leftTime > 0:
            self.leftTime -= 1
        return False

    # 从inPort或返回道中取一个车身，并运送到某个车道
    # 若返回False，则说明此指令还不能完成，返回道上的车身还没有到达
    def pickup(self, source, targetId):
        if source == 'inPort':
            self.carry = self.inPort.getFrontCar()
            self.leftTime = self.loadCost[targetId]
            self.target = targetId
            self.idle = False
            return True
        elif source == 'bPath':
            self.carry = self.Paths.getFront(-1)
            if self.carry is None:
                return False
            # 瞬间移动到返回道取下车身，剩下的时间移动到目标位置
            self.Paths.unloadCar(-1)
            self.leftTime = self.transCost[targetId]
            self.target = targetId
            self.idle = False
            return True

    # idle=False, leftTime=0时调用
    def arrive(self):
        if not self.Paths.loadCar(self.carry, self.target):
            return False
        self.idle = True
        self.target = None
        self.carry = None
        return True


class rMover(object):
    def __init__(self, outPort, Paths):
        self.outPort = outPort  # 指向连接的输出口
        self.Paths = Paths  # 指向连接的车道组
        self.idle = True  # 是否空闲
        self.carry = None  # 是否装着那辆车
        self.target = None  # 是否在往哪个车道走
        self.leftTime = 0  # 如果在移动，还需要多久
        self.loadCost = [18, 12, 6, 0, 12, 18]  # 将车身从进车道送入出口的耗时
        self.transCost = [24, 18, 12, 6, 12, 18]  # 将车身从进车道接下再送入返回道的耗时
        self.wait = 0

    def execute(self, source, target):
        # Env每一秒都向Mover调用该函数
        # 返回True代表这条指令被完成
        # 指令格式：[pathid] ['outPort'/'bPath'],
        if target == 'wait':
            if self.idle:
                self.idle = False
                self.wait = self.loadCost[source] - 1
                return False
            else:
                self.wait -= 1
                if self.wait != 0:
                    return False
                else:
                    self.idle = True
                    return True
        if self.idle:
            self.pickup(source, target)
        if not self.idle and self.leftTime == 0:
            if self.arrive():
                return True
        if self.leftTime > 0:
            self.leftTime -= 1
        return False

    def pickup(self, source, target):
        # source is a pathid
        # 要取的车身还没到就不去
        self.carry = self.Paths.getFront(source)
        if self.carry is None:
            return False
        # 瞬间到达相应位置取下车身，剩下的时间移动到目标位置
        self.Paths.unloadCar(source)
        if target == 'outPort':
            self.leftTime = self.loadCost[source]
            self.target = source
        # otherwise target is bPath
        elif target == 'bPath':
            self.leftTime = self.transCost[source]
            self.target = -1
        self.idle = False
        return True

    # idle=False, leftTime=0时调用
    def arrive(self):
        if self.target == -1:
            # 等待返回道不堵塞才放
            if not self.Paths.loadCar(self.carry, self.target):
                return False
        else:
            # 如果目标是出口，则不可能会堵塞
            self.outPort.enter(self.carry)
        self.idle = True
        self.target = None
        self.carry = None
        return True


class env(object):
    def __init__(self, sourcePath):
        self.sourcePath = sourcePath
        self.carPool = self.getCarList()
        self.backup = []
        for car in self.carPool:
            self.backup.append(car)
        # devices
        self.inPort = inPort(self.carPool)
        self.outPort = outPort(len(self.carPool))
        self.Paths = Paths()
        self.lMover = lMover(self.inPort, self.Paths)
        self.rMover = rMover(self.outPort, self.Paths)
        # parameter
        self.time = 0
        self.bCnt = 0
        self.locations = [[] for _ in range(len(self.carPool))]
        # data
        self.data = []
        self.scores = []
        self.times = 0
        self.bcnts = []
        self.bcntSum = 0

    def run_q1(self, optList):
        lopts = optList
        ropts = []
        for _ in range(50000):
            if self.outPort.finished():
                break
            self.Paths.moveCars()
            if self.Paths.bPath[9] is not None:
                # 如果现在正在做任务，那么取返回道的车身就是下一个任务，否则立刻做
                if self.lMover.idle and (len(lopts) == 0 or lopts[0][0] != 'bPath'):
                    lopts.insert(0, ['bPath', random.randint(0, 5)])
                    print('有车到达！,t=', self.time)
                elif not self.lMover.idle and (len(lopts) == 1 or lopts[1][0] != 'bPath'):
                    lopts.insert(1, ['bPath', random.randint(0, 5)])
                    print('有车到达！,t=', self.time)
                # print(lopts)
            if len(lopts) != 0:
                lopt = lopts[0]
                # print(lopts, self.lMover.carry, len(self.inPort.carList), len(self.outPort.sequence))
                if self.lMover.execute(lopt[0], lopt[1]):
                    lopts.pop(0)
                    # ropts.append([lopt[1], 'outPort'])
                    # 随机测试返回道功能
                    if random.randint(0, 9) == 9:
                        # ropts.append([lopt[1], 'outPort'])
                        ropts.append([lopt[1], 'bPath'])
                        self.bCnt += 1
                    else:
                        ropts.append([lopt[1], 'outPort'])
            if len(ropts) != 0:
                ropt = ropts[0]
                if self.rMover.execute(ropt[0], ropt[1]):
                    ropts.pop(0)
            # ----
            self.getLocations()
            # print(self.Paths.bPath)
            # print(lopts, '   ', ropts)
            self.time += 1
            '''self.Paths.getInfo()
            if _ % 100 == 0:
                os.system("pause")'''
        print('num=', len(self.outPort.sequence))
        print(self.time)
        self.getAnsExl()

    def run_q2(self, optList, takeList):
        # 分别给出lMover的指令与rMover的指令
        lopts = []
        for opt in optList:
            lopts.append(opt)
            if opt[1] != 3:
                lopts.append(['wait', opt[1]])
        ropts = []
        for opt in takeList:
            if opt[1] == 'bPath':
                self.bCnt += 1
            if opt[0] != 3:
                ropts.append([opt[0], 'wait'])
            ropts.append(opt)

        for _ in range(60000):
            if self.outPort.finished():
                break
            self.Paths.moveCars()
            if len(lopts) != 0:
                lopt = lopts[0]
                if self.lMover.execute(lopt[0], lopt[1]):
                    lopts.pop(0)
            if len(ropts) != 0:
                ropt = ropts[0]
                if self.rMover.execute(ropt[0], ropt[1]):
                    ropts.pop(0)
            self.getLocations()
            self.time += 1
        print('num=', len(self.outPort.sequence))
        print(self.time)
        print('score=', self.getScore())

    def getScore(self):
        seq = self.outPort.sequence
        # ----
        reward1 = 100
        fuelNum = 0
        flag = 0
        for car in seq:
            # 如果是混动车身
            if car.powerType == 2:
                if flag and fuelNum != 2:
                    reward1 -= 1
                fuelNum = 0
                flag = 1
            else:
                fuelNum += 1
        # ----
        reward2 = 100
        nums = [0, 0]
        headType = seq[0].driverType // 4
        tailType = (headType + 1) % 2
        for car in seq:
            # print(car.driverType, end=' ')
            if (car.driverType // 4) == headType and nums[tailType] != 0:
                if nums[0] != nums[1]:
                    reward2 -= 1
                nums = [0, 0]
            else:
                nums[car.driverType // 4] += 1
        # ----
        reward3 = 100 - self.bCnt
        reward4 = 100 - 0.01 * (self.time - 9 * len(self.backup) - 72)
        total = reward1 * 0.4 + reward2 * 0.3 + reward3 * 0.2 + reward4 * 0.1
        print(reward1 * 0.4, reward2 * 0.3, reward3 * 0.2, reward4 * 0.1)
        return total

    def getLocations(self):
        for car in self.backup:
            index = int(car.Id) - 1
            self.locations[index].append(car.getLocation())

    def getAnsExl(self):
        with open('output.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i for i in range(self.time)])
            Id = 0
            for line in self.locations:
                Id += 1
                writer.writerow(line)
        file = pd.read_csv('output.csv', encoding='utf-8')
        file.to_excel('output.xlsx', sheet_name='result')

    def getCarList(self):
        carList = []
        wb = xlrd.open_workbook(self.sourcePath)
        sheet = wb.sheet_by_index(0)
        for i in range(1, sheet.nrows):
            Id = int(sheet.row(i)[0].value)
            modelType = sheet.row(i)[1].value
            powerType = sheet.row(i)[2].value
            driverType = sheet.row(i)[3].value
            if powerType == 'fuel':
                powerType = 1
            else:
                powerType = 2
            if driverType == 'two wheel':
                driverType = 2
            else:
                driverType = 4
            # --------
            carList.append(car(Id, modelType, powerType, driverType))
        return carList

    def getSeqFromRes(self, offList):
        alist = []
        bCnt = 0
        for i in range(len(offList) + 1):
            # [index, id]
            alist.append([i, i])
        for i in range(len(offList)):
            if offList[i] > 1:
                bCnt += 1
            if offList[i] != 0:
                alist[i][0] += 1
            alist[i][0] += offList[i]
        alist.sort()
        seq = []
        for index, Id in alist:
            seq.append(self.backup[Id])
        return seq, bCnt

    def getSeqScore(self, seqList, bCnt):
        seq = seqList
        # ----
        reward1 = 100
        fuelNum = 0
        flag = 0
        for car in seq:
            # 如果是混动车身
            if car.powerType == 2:
                if flag and fuelNum != 2:
                    reward1 -= 1
                fuelNum = 0
                flag = 1
            else:
                fuelNum += 1
        # ----
        reward2 = 100
        nums = [0, 0]
        headType = seq[0].driverType // 4
        tailType = (headType + 1) % 2
        for car in seq:
            # print(car.driverType, end=' ')
            if (car.driverType // 4) == headType and nums[tailType] != 0:
                if nums[0] != nums[1]:
                    reward2 -= 1
                nums = [0, 0]
            else:
                nums[car.driverType // 4] += 1
        # ----
        reward3 = 100 - bCnt
        total = reward1 * 0.4 + reward2 * 0.3 + reward3 * 0.2
        # print(total, reward1 * 0.4, reward2 * 0.3, reward3 * 0.2, bCnt)
        return total

    def gaList2offList(self, gaList):
        offList = []
        for value in gaList:
            if int(value) <= 300:
                offList.append(0)
            elif int(value) <= 450:
                offList.append(1)
            else:
                offList.append(int(value) - 450)
        return offList

    def cost(self, gaList):
        offList = self.gaList2offList(gaList)
        seq, bCnt = self.getSeqFromRes(offList)
        self.bcntSum += bCnt
        res = self.getSeqScore(seq, bCnt)
        return -res

    def iterSeq(self, pops=6000, iters=40, mut=0.01):
        nums = len(self.backup)
        lowerBound = [0] * (nums - 1)
        upperBound = [(nums - i) for i in range(nums - 1)]
        ga = GA.GA(func=self.cost, n_dim=nums - 1, size_pop=pops, max_iter=iters, lb=lowerBound, ub=upperBound,
                   prob_mut=mut, precision=1)
        offlist, best_reward = ga.run()
        print(self.gaList2offList(offlist), -best_reward)
        return self.gaList2offList(offlist), -best_reward


def getOrderBySeq(seq, offlist):
    # 获取lMover与rMover的操作序列
    # initialization
    seqIndex = 0
    offlist.append(0)
    deliverCnt = [1] * len(seq)
    lMoverOpt = []
    rMoverOpt = []
    pathid = [-1] * len(seq)
    # cleanTheOfflist
    for i in range(len(offlist) - 1):
        if offlist[i] == 1 and offlist[i + 1] != 0:
            offlist[i] = 0
    # generate
    index = 0
    while index < len(seq):
        if offlist[index] == 0:
            lMoverOpt.append(['inPort', 2, index])
            rMoverOpt.append([2, 'outPort', index])
        elif offlist[index] > 1:
            lMoverOpt.append(['inPort', 2, index])
            rMoverOpt.append([2, 'bPath', index])
        else:
            lMoverOpt.append(['inPort', 3, index])
            lMoverOpt.append(['inPort', 2, index + 1])
            rMoverOpt.append([2, 'outPort', index + 1])
            rMoverOpt.append([3, 'outPort', index])
            index += 1
        index += 1
    print(lMoverOpt, rMoverOpt)
    # insert
    optIndex = 0
    finalCheck = []
    for res in seq:
        finalCheck.append(res.Id)
    while True:
        if len(finalCheck) == 0:
            break
        while rMoverOpt[optIndex][1] == 'bPath':
            optIndex += 1
        carId = finalCheck.pop(0)

        if rMoverOpt[optIndex][2] + 1 != carId:
            print(carId)
            beforeId = rMoverOpt[optIndex - 1][2]
            rMoverOpt.insert(optIndex, [2, 'outPort', carId - 1])
            for i in range(len(lMoverOpt) - 1, -1, -1):
                if lMoverOpt[i][2] == beforeId:
                    lMoverOpt.insert(i + 1, ['bPath', 2, carId])
                    print('yes', beforeId, carId)
        optIndex += 1
    # lastCheck
    print('lastCheck')
    optIndex = 0
    finalCheck = []
    for res in seq:
        finalCheck.append(res.Id)
    while True:
        if len(finalCheck) == 0:
            break
        while rMoverOpt[optIndex][1] == 'bPath':
            optIndex += 1
        carId = finalCheck.pop(0)
        if rMoverOpt[optIndex][2] + 1 != carId:
            print(carId)
        optIndex += 1
    return lMoverOpt, rMoverOpt


def checkList(olist):
    nextL = 0
    for i in range(len(olist)):
        if olist[i] > 1:
            if i + olist[i] >= nextL:
                nextL = i + olist[i]
            else:
                return False
    return True


if __name__ == '__main__':
    e = env('D1.xlsx')
    e.iterSeq(iters=100)
    e2 = env('D2.xlsx')
    e2.iterSeq(iters=100)