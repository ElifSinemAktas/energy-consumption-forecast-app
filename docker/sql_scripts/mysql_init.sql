CREATE USER 'gitea'@'%' IDENTIFIED BY 'gitea';
CREATE DATABASE IF NOT EXISTS `gitea`;
GRANT ALL PRIVILEGES ON `gitea` . * TO 'gitea'@'%';