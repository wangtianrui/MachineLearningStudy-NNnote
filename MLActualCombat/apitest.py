import threading
import numpy as np

balance = 0
lock = threading.Lock()


def testThreadingLock():
    def change_it(n):
        """
        存钱
        :param n:
        :return:
        """
        global balance
        balance = balance + n
        balance = balance - n
        print(balance)

    def run_thread(n):
        for i in range(10000):
            lock.acquire()
            try:
                change_it(n)
            finally:
                lock.release()

    t1 = threading.Thread(target=run_thread, args=(5,))
    t2 = threading.Thread(target=run_thread, args=(8,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__ == "__main__":
    # testThreadingLock()
    array1 = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    array3 = np.array([[0], [1], [2], [3], [4]])
    array2 = (3 == array1).astype(int)
    print(array1 * array1)
    print(array2)
