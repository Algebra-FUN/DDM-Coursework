# A short report of Web3

When people talk about Web3, they will mention the terminologies like bit-coins,  meta-universe and the chance to get rich. In my opinion, I think the key point of Web3 is decentralization, not the fancy word like meta-universe which only involves some already existed technologies, such as VR and AR. 

### The history of Web

If we want to understand the concept about web3, we should first understand the concept of  web1 and web2.

**Web1** just refers the first generation of web, at that time, the web is static, which means that it is just for professional user to create content and give a way to let others to view via Internet. It was not easy and expensive for a common user to host a website and share their creations.

**Web2** refers to the web that the big companies host server as platform to let user create contents and they can utilize these contents to get benefit. Although users can easily shares their creations via the platform, but they don't own data since the data is on the remote servers. The Web2 is centralized, so this kind of  Internet is controlled by some big companies and all the users' data are controlled by them, which means what you viewed, what you brought and what you chatted, they all know.

**Web3** is aimed to create decentralized web with modern decentralization technologies such as block-chain, NFT, IPFS. The content creators can claim their ownership and own their data instead of the platform own these data. The block-chain is the most important technology in them, since it provide a decentralized, secure and non-reversible method to record history(like transaction history), which method is created new brand of digital currency ---- bit-coin.

<img src="https://qph.cf2.quoracdn.net/main-qimg-c310aa4c6c44f698d4548b29117e295b-lq" style="zoom: 80%;" />

### Block-chain

When people talk about block-chain, they always equate it with bit-coin, however they are not the same concept. Bit-coin is just derivative of block-chain technology.

First of all, let's brief introduce the block-chain. The key to understand block-chain is not to understand the technical detail of the implementation, is to understand why people need it and why design it in this way.

Since people need a decentralized system to record data non-reversibly, such as transaction data. The most of naive decentralized system has the security problem that someone may manipulate record data and fraud other. So people create centralized system to host a trusted third party to maintain these record data, such as bank in the finance system.  But people still want freedom and to get rid of these third party which is just partly trusted(like Federal Reserve System of US, they just print green paper to rob resources from all over the world). 

So how can we both ensure decentralized and security(data non-reversible)?  A genius invented block-chain to provide the solution. This decentralized system based on cryptographic proof instead of trust, which means the computation power replace the role of the third party. This system is secure since the record data on the block-chain is computationally impractical to reverse which would protect the data from being manipulated, which depends on the attackers can't obtain sufficient computation power more than half to carry out an attack.

So the problem converts to how can we prevent attackers take half computation power? The bit-coin comes out to solve this problem. Since this block-chain system depends on the honest computation power, so we need a lot, but how to persuade to contribute their computation source to our block-chain system in order to ensure the data security? The truth is that people have no reason to contribute to make world better unless they can get benefit. So one can invent a kind of digital money to seduce people to come to "mining" with their computational source.

<img src="https://www.edureka.co/blog/wp-content/uploads/2019/03/mining-pool-blockchain-mining-edureka.png" alt="Blockchain Mining- All you need to know | Edureka" style="zoom:50%;" />

#### Application of Block-Chain

There are many application of block-chain technology. The most significant one is the peer-to-peer electronic cash system(such as bit-coin), which uses block-chain to record transaction history and computational progress of cryptographic proof to publish the digital currency. 

Digital asset management system uses block-chain to record the ownership of these digital asset and the transaction records of the ownership. In this way, people can claim the ownership of some digital data.

Apps use the decentralized technologies, called Dapps. A **decentralised application** is an app that can operate autonomously that run on a decentralized computing, blockchain or other distributed ledger system. Like traditional applications, DApps provide some function or utility to its users. However, unlike traditional applications, DApps operate without human intervention and are not owned by any one entity, rather DApps distribute tokens that represent ownership.

<img src="https://miro.medium.com/max/1400/1*L5ApbOvu0Pf-oRXFYd3vkA.png" alt="What is a DAPP? | by Shaan Ray | Towards Data Science" style="zoom: 33%;" />

### IPFS

In the Web3, we also need other technologies or infrastructures to implement decentralize Internet in the real world(only in the application layer, since in the real physical layer of network, the network cables are controlled by the providers company and government). We need a distributed system to store and access data in a decentralized manner.

IPFS is one of the distributed system for storing and accessing files, websites, applications, and data. 

In the Web2, we should know the URL(*Uniform Resource Locator*) to find the data we want and download it, which means the data is located at one specified server and meanwhile controlled by it. If a Great Fire Wall block this url or this server, one can access this data, which is the disadvantages of centralized Internet. 

The decentralized system can solve this problem. In IPFS, you only need the CID(*content identifier*) to get the data you want. Your computer uses IPFS to ask lots of computers around the world to share the data with you, meanwhile utilize your device to distribute this resource.

![A simplified IPFS network and its components. A user adds data to the... |  Download Scientific Diagram](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRad2ICeWUoKCd22Ix2jXe52bXHMQzr_nimOipHuRCzPulPyA5lVuBFta2tRE8IoRLndFo&usqp=CAU)

#### The advantage of this kind of decentralized system

Making it possible to download a file from many locations that aren't managed by one organization:

- **Supports a resilient internet.** If someone attacks Wikipedia's web servers or an engineer at Wikipedia makes a big mistake that causes their servers to catch fire, you can still get the same webpages from somewhere else.
- **Makes it harder to censor content.** Because files on IPFS can come from many places, it's harder for anyone (whether they're states, corporations, or someone else) to block things. We hope IPFS can help provide ways to circumvent actions like these when they happen.
- **Can speed up the web when you're far away or disconnected.** If you can retrieve a file from someone nearby instead of hundreds or thousands of miles away, you can often get it faster. This is especially valuable if your community is networked locally but doesn't have a good connection to the wider internet. 