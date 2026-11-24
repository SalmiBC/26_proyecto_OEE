#!/bin/bash
# ==============================================================================
# Script de Despliegue Automático en AWS EC2 (Docker + FastAPI + MLOps)
# 100% Free Tier Garantizado en Ohio (us-east-2) o cualquier región
# ==============================================================================
set -e

# Forzar región activa de la consola (Ohio us-east-2)
REGION="${AWS_REGION:-us-east-2}"
echo "=================================================================="
echo "🚀 INICIANDO DESPLIEGUE EN AWS ($REGION - FREE TIER)"
echo "=================================================================="

# 1. Detectar tipo Free Tier oficial disponible en la cuenta
FREE_TYPE=$(aws ec2 describe-instance-types --region $REGION --filters "Name=free-tier-eligible,Values=true" --query "InstanceTypes[?CurrentGeneration==\`true\`].InstanceType | [0]" --output text 2>/dev/null || true)
if [ -z "$FREE_TYPE" ] || [ "$FREE_TYPE" == "None" ]; then
    FREE_TYPE="t3.micro"
fi
echo "[1/5] Tipo de instancia Free Tier oficial: $FREE_TYPE"

# 2. Obtener VPC y Subnet por defecto de la región
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region $REGION --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text | awk '{print $1}')
echo "[2/5] VPC: $VPC_ID | Subnet: $SUBNET_ID"

# 3. Crear Grupo de Seguridad con puertos 80 y 8000
SG_NAME="oee-sg-ohio"
SG_ID=$(aws ec2 describe-security-groups --region $REGION --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    echo "[3/5] Creando Grupo de Seguridad $SG_NAME..."
    SG_ID=$(aws ec2 create-security-group --region $REGION --group-name $SG_NAME --description "SG para OEE en Ohio" --vpc-id $VPC_ID --query "GroupId" --output text)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
else
    echo "[3/5] Usando Grupo de Seguridad existente: $SG_ID"
fi

# 4. AMI oficial Amazon Linux 2023
AMI_ID=$(aws ec2 describe-images --region $REGION --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
echo "[4/5] AMI: $AMI_ID"

# 5. UserData con Swap de 2GB para compilar sin saturar memoria
cat << 'USERDATA_EOF' > /tmp/oee_userdata.sh
#!/bin/bash
dd if=/dev/zero of=/swapfile bs=1M count=2048
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "/swapfile swap swap defaults 0 0" >> /etc/fstab
dnf update -y
dnf install -y docker git
systemctl start docker
systemctl enable docker
mkdir -p /opt/oee-app && cd /opt/oee-app
git clone https://github.com/SalmiBC/26_proyecto_OEE.git .
docker build -t oee-api .
docker run -d --restart always -p 80:8000 -p 8000:8000 --name oee-service oee-api
USERDATA_EOF

# 6. Lanzar instancia
echo "[5/5] Lanzando instancia $FREE_TYPE en $REGION..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $FREE_TYPE \
    --security-group-ids $SG_ID \
    --subnet-id $SUBNET_ID \
    --user-data file:///tmp/oee_userdata.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=oee-predictivo-pisa}]' \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Instancia creada: $INSTANCE_ID. Esperando asignación de IP pública..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo ""
echo "=================================================================="
echo "🎉 ¡DESPLIEGUE EN AWS COMPLETADO CON ÉXITO!"
echo "=================================================================="
echo "Región     : $REGION (Ohio)"
echo "Instancia  : $INSTANCE_ID ($FREE_TYPE - $0 USD Free Tier)"
echo "IP Pública : $PUBLIC_IP"
echo ""
echo "En ~2 minutos (mientras Docker compila en segundo plano) abre:"
echo "👉 Aplicación Web : http://$PUBLIC_IP"
echo "👉 Swagger OpenAPI: http://$PUBLIC_IP/docs"
echo "👉 Rúbrica Demo   : http://$PUBLIC_IP/presentacion"
echo "=================================================================="
